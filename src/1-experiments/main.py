import argparse
import yaml
import os
import sys
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
        return 500

    def __getitem__(self, idx):
        params = self.config['parameters']
        omega = params.get('omega', 64.0)
        
        # --- PML Settings for Experiment Arm A ---
        # Prioritizes YAML values for s1_pml_strategy study
        npml = params.get('npml', 112) 
        pml_strategy = params.get('pml_strategy', 'hybrid')
        custom_eta = params.get('eta', 50.0)
        
        # Generate the specific PML profile for this experiment run
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
        fourier_channels = self.encoder.encode(x, y, omega)
        
        # --- Meta-Channels ---
        omega_tensor = torch.full((1, self.grid_size, self.grid_size), omega / 128.0, device=self.device)
        direction = 1.0 if params.get('direction', 'up') == 'up' else -1.0
        dir_tensor = torch.full((1, self.grid_size, self.grid_size), direction, device=self.device)

        # Total Stack (26 Channels)
        input_stack = torch.cat([
            rhs.unsqueeze(0), pml_map.unsqueeze(0), fourier_channels, 
            omega_tensor, dir_tensor, x.unsqueeze(0), y.unsqueeze(0)
        ], dim=0)
        
        target = torch.randn(2, self.grid_size, self.grid_size, device=self.device) 
        return input_stack, target

def run_experiment(config_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Device: {device} ---")

    full_config_path = os.path.abspath(os.path.join(base_path, config_path))
    with open(full_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Output directory for Arm A results
    results_dir = os.path.join(os.path.dirname(__file__), config['experiment_id'])
    os.makedirs(os.path.join(results_dir, "weights"), exist_ok=True)
    
    pml_gen = pml_mod.PMLManager(grid_size=512)
    src_gen = src_mod.SourceFactory(grid_size=512)
    encoder = enc_mod.FourierEncoder(num_frequencies=10)
    
    dataset = WaveDataset(config, pml_gen, src_gen, encoder, device)
    loader = DataLoader(dataset, batch_size=1)

    # Dynamic Channel Detection (fixes the 25 vs 26 error)
    sample_input, _ = dataset[0]
    in_channels = sample_input.shape[0]
    
    model = cnn_mod.FlatOperator(in_channels=in_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['parameters'].get('learning_rate', 0.00011))
    criterion = torch.nn.MSELoss()

    print(f"--- Launching {config['experiment_id']} with PML Strategy: {config['parameters'].get('pml_strategy')} ---")

    for epoch in range(config['parameters'].get('epochs', 1000)):
        model.train()
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.8f}")
            torch.save(model.state_dict(), f"{results_dir}/weights/latest.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    run_experiment(args.config)