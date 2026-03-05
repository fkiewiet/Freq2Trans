import argparse
import yaml
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import importlib.util

# --- Handle Pathing ---
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def import_from_path(module_name, relative_path):
    file_path = os.path.join(base_path, relative_path)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load Core Physics Modules
pml_mod = import_from_path("pml_manager", "1-core/pml_manager.py")
src_mod = import_from_path("source_factory", "1-core/source_factory.py")
enc_mod = import_from_path("encoding", "1-core/encoding.py")
cnn_mod = import_from_path("CNN_operator", "1-models/CNN_operator.py")

class WaveDataset(Dataset):
    def __init__(self, config, pml_gen, src_gen, encoder, device):
        self.config = config
        self.pml_gen = pml_gen
        self.src_gen = src_gen
        self.encoder = encoder
        self.device = device
        self.grid_size = 512

    def __len__(self):
        return self.config['parameters'].get('epoch_size', 500)

    def __getitem__(self, idx):
        params = self.config['parameters']
        omega = params.get('omega', 64.0)
        
        # --- Default Physics Settings ---
        npml = params.get('npml', 112) 
        pml_strategy = params.get('pml_strategy', 'standard')
        custom_eta = params.get('eta', 50.0)
        
        # Generate the specific PML profile
        eta = self.pml_gen.get_eta_value(omega, pml_strategy, custom_eta=custom_eta)
        pml_map = self.pml_gen.generate_2d_pml(npml, eta).to(self.device)
        
        # --- Source Generation ---
        rhs, _ = self.src_gen.create_source(npml, mode=params.get('source_type', 'point'))
        rhs = rhs.to(self.device)
        
        # --- Spatial Encoding ---
        y, x = torch.meshgrid(
            torch.linspace(0, 1, self.grid_size, device=self.device), 
            torch.linspace(0, 1, self.grid_size, device=self.device), 
            indexing='ij'
        )
        # encoder.encode (num_frequencies=10) returns 20 channels
        fourier_channels = self.encoder.encode(x, y, omega)
        
        # --- Meta-Channels & Directional Instruction ---
        omega_tensor = torch.full((1, self.grid_size, self.grid_size), omega / 128.0, device=self.device)
        
        # Instruction Bit: +1.0 for 'up', -1.0 for 'down'
        direction_str = params.get('direction', 'up')
        dir_val = 1.0 if direction_str == 'up' else -1.0
        dir_tensor = torch.full((1, self.grid_size, self.grid_size), dir_val, device=self.device)

        # Total Stack: 1(RHS) + 1(PML) + 20(Fourier) + 1(Omega) + 1(Dir) + 2(Coords) = 26
        input_stack = torch.cat([
            rhs.unsqueeze(0), 
            pml_map.unsqueeze(0), 
            fourier_channels, 
            omega_tensor, 
            dir_tensor, 
            x.unsqueeze(0), 
            y.unsqueeze(0)
        ], dim=0)
        
        # Target placeholder (Real/Imaginary components)
        target = torch.randn(2, self.grid_size, self.grid_size, device=self.device) 
        return input_stack, target

def run_experiment(config_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on: {device} ---")

    full_config_path = os.path.abspath(os.path.join(base_path, config_path))
    with open(full_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Isolated Results Directory
    results_dir = os.path.join(base_path, "results", config['experiment_id'])
    os.makedirs(os.path.join(results_dir, "weights"), exist_ok=True)
    
    pml_gen = pml_mod.PMLManager(grid_size=512)
    src_gen = src_mod.SourceFactory(grid_size=512)
    encoder = enc_mod.FourierEncoder(num_frequencies=10)
    
    dataset = WaveDataset(config, pml_gen, src_gen, encoder, device)
    loader = DataLoader(dataset, batch_size=config['parameters'].get('batch_size', 1), shuffle=True)

    # --- Dynamic Channel Detection ---
    sample_input, _ = dataset[0]
    in_channels = sample_input.shape[0]
    print(f"--- Detected {in_channels} input channels ---")
    
    model = cnn_mod.FlatOperator(in_channels=in_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(config['parameters'].get('learning_rate', 0.00011)))
    criterion = torch.nn.MSELoss()

    print(f"--- Launching Study Arm C: {config['experiment_id']} ---")

    try:
        epochs = config['parameters'].get('epochs', 2000)
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for inputs, targets in loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % 10 == 0:
                avg_loss = epoch_loss / len(loader)
                print(f"Epoch {epoch} | Avg Loss: {avg_loss:.8e}")
                torch.save(model.state_dict(), f"{results_dir}/weights/latest.pt")
                
    except KeyboardInterrupt:
        print("\nInterrupt detected. Saving weights to interrupted.pt...")
        torch.save(model.state_dict(), f"{results_dir}/weights/interrupted.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    run_experiment(args.config)