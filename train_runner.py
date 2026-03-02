import torch
import numpy as np
from pathlib import Path

# Imports from your project structure
from src.algorithm.dataset_handler import DatasetHandler
from src.algorithm.transfer import train_operator
from src.models.cnn import PlainCNN  # Ensure this matches your file name

def main():
    # 1. Configuration: Using your saved parameters
    cfg = {
        "run_id": "wave_transfer_v1",
        "npml": 104,              # As per your saved info
        "eta": 70.0,               # As per your saved info
        #"br_mean_expected": 0.6999, # Target metric
        "lr": 1e-3,
        "epochs": 100,
        "batch_size": 16,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    # 2. Path Management
    # Data is organized by physics params so you don't mix up eta=4 with eta=2
    data_dir = Path(f"data_cache/npml{cfg['npml']}_eta{int(cfg['eta'])}")
    run_dir = Path(f"experiments/runs/{cfg['run_id']}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # 3. Dataset Handling (Load once, reuse forever)
    handler = DatasetHandler(data_dir)
    try:
        print(f"Checking for cached data in {data_dir}...")
        train_data = handler.load_split("train")
        val_data = handler.load_split("val")
        print("Successfully loaded cached dataset.")
    except FileNotFoundError:
        print("No cache found. You need to run your dataset generation script first.")
        # Alternatively, call your generation function here:
        # train_data, val_data = generate_my_data(cfg)
        return

    # 4. Model Initialization
    # We initialize the model here so we can swap it easily (e.g., to a UNet)
    model = PlainCNN().to(cfg['device'])

    # 5. Training execution
    # This calls the complete transfer.py function we just built
    trained_model, history = train_operator(model, train_data, val_data, cfg)

    # 6. Save the "Reload-Friendly" Artifact
    checkpoint_path = run_dir / "model_checkpoint.pth"
    torch.save({
        "state_dict": trained_model.state_dict(),
        "history": history,  # Needed for your loss curve graph
        "config": cfg,       # Stores npml, eta, and expected br_mean
        "status": "completed"
    }, checkpoint_path)

    print(f"Run Finished. Checkpoint saved to: {checkpoint_path}")

if __name__ == "__main__":
    main()