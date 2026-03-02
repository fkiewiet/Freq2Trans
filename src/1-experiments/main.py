import argparse
import yaml
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import importlib.util

# --- Handle Pathing for Folders with Numbers ---
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Dynamically load your physics and model files
pml_mod = import_from_path("pml_manager", os.path.join(base_path, "1-core/pml_manager.py"))
src_mod = import_from_path("source_factory", os.path.join(base_path, "1-core/source_factory.py"))
enc_mod = import_from_path("encoding", os.path.join(base_path, "1-core/encoding.py"))
cnn_mod = import_from_path("CNN_operator", os.path.join(base_path, "1-models/CNN_operator.py"))

class WaveDataset(Dataset):
    def __init__(self, config, pml_gen, src_gen, encoder):
        self.config = config
        self.pml_gen = pml_gen
        self.src_gen = src_gen
        self.encoder = encoder
        self.grid_size = 512

    def __len__(self):
        return 500

    def __getitem__(self, idx):
        params = self.config['parameters']
        omega = params.get('omega', 64.0)
        npml = params.get('npml', 112)
        
        eta = self.pml_gen.get_eta_value(omega, params.get('pml_strategy', 'hybrid'), custom_eta=npml)
        pml_map = self.pml_gen.generate_2d_pml(npml, eta)
        rhs, _ = self.src_gen.create_source(npml, mode=params.get('source_type', 'point'))
        
        y, x = torch.meshgrid(torch.linspace(0, 1, 512), torch.linspace(0, 1, 512), indexing='ij')
        fourier_channels = self.encoder.encode(x.to(pml_map.device), y.to(pml_map.device), omega)
        
        omega_tensor = torch.full((1, 512, 512), omega / 128.0).to(pml_map.device)
        direction = 1.0 if params.get('direction', 'up') == 'up' else -1.0
        dir_tensor = torch.full((1, 512, 512), direction).to(pml_map.device)

        input_stack = torch.cat([
            rhs.unsqueeze(0), pml_map.unsqueeze(0), fourier_channels, 
            omega_tensor, dir_tensor, x.unsqueeze(0).to(pml_map.device), y.unsqueeze(0).to(pml_map.device)
        ], dim=0)
        
        target = torch.randn(2, 512, 512).to(pml_map.device) 
        return input_stack, target

def run_experiment(config_path):
    full_config_path = os.path.abspath(os.path.join(base_path, config_path))
    with open(full_config_path, 'r') as f:
        config = yaml.safe_load(f)

    results_dir = os.path.join(os.path.dirname(__file__), config['experiment_id'])
    os.makedirs(os.path.join(results_dir, "weights"), exist_ok=True)
    
    pml_gen = pml_mod.PMLManager(grid_size=512)
    src_gen = src_mod.SourceFactory(grid_size=512)
    encoder = enc_mod.FourierEncoder(num_frequencies=10)
    
    model = cnn_mod.FlatOperator(in_channels=25).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=config['parameters'].get('learning_rate', 0.00011))
    criterion = torch.nn.MSELoss()

    loader = DataLoader(WaveDataset(config, pml_gen, src_gen, encoder), batch_size=1)

    for epoch in range(config['parameters'].get('epochs', 1000)):
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")
            torch.save(model.state_dict(), f"{results_dir}/weights/latest.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    run_experiment(args.config)