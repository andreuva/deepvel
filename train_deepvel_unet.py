import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from glob import glob
import muram  # Assumes the muram module is available and in PYTHONPATH
from deepvel_unet2 import UDeepVel
from dataset import MURAMVelocityDataset
import argparse, os, json
from tqdm import tqdm
import time
import wandb

# Main training script
parser = argparse.ArgumentParser(description='Train neural network')
parser.add_argument('--config', '--config', default='config.json', type=str, metavar='CONF', help='Configuration file')
parsed = vars(parser.parse_args())

# Load the config file
with open(parsed["config"], 'r') as f:
    config = json.load(f)

if not os.path.exists(config['training']['save_path']):
    os.makedirs(config['training']['save_path'])

# Initialize wandb
# Initialize wandb
wandb.init(project="velocity_prediction",
    config=config,
    # name with the basic config + current time and date
    name=f"deepvel_unet2_imsize_{config['dataset']['patch_size']}_aug_{config['dataset']['aug']}_lr_{config['training']['learning_rate']}_t_{time.strftime('%Y%m%d-%H%M%S')}",
)

# Load data from files (assumes the data folder exists and muram functions work)
intensity_files = sorted(glob(f'{config["dataset"]["data_path"]}/I_out*'))
iterations = [int(file[-6:]) for file in intensity_files]

Intensities = []
velocities = []
times = []
for iter in tqdm(iterations, desc="Loading the data from MuRam files"):
    # Load the intensity data
    Intensities.append(muram.MuramIntensity(config["dataset"]["data_path"], iter))
    # Load the velocity data from tau_slice (using slice=1 as before)
    tau_slice = muram.MuramTauSlice(config["dataset"]["data_path"], iter, 1)
    vx, vy = tau_slice.vy, tau_slice.vz
    velocities.append((vx, vy))
    times.append(tau_slice.time)

# Create the dataset and dataloader
dataset = MURAMVelocityDataset(Intensities, velocities, times,
                                img_size=config["dataset"]["patch_size"],
                                batch_size=config["dataset"]["batch_size"],
                                aug=config["dataset"]["aug"],
                                seed=config["dataset"]["seed"],)
dataloader = DataLoader(dataset,
                        batch_size=None,
                        num_workers=config["dataset"]["num_workers"])

# Comprobar la GPU
print('#'*70)
print('CUDA INFO')
print(f'Cuda available: {torch.cuda.is_available()}')
print(f'Number of cuda devices: {torch.cuda.device_count()}')
print(f'Selecting GPU:', config["training"]["used_gpu"])
print(f'Current selected device: {torch.cuda.current_device()}')
print(f'Name of the used device : {torch.cuda.get_device_name(config["training"]["used_gpu"])}')
print(f'Info of the device : {torch.cuda.get_device_properties(config["training"]["used_gpu"])}')
print('#'*70)

# Create model, loss, and optimizer
model = UDeepVel(
    input_channels=config["model"]["input_channels"],   # Three consecutive intensity images.
    output_channels=config["model"]["output_channels"], 
    n_latent_chanels=config["model"]["n_latent_chanels"],
    chanels_multiples=config["model"]["chanels_multiples"],
    is_attn_encoder=config["model"]["is_attn_encoder"],
    is_attn_decoder=config["model"]["is_attn_decoder"],
    n_blocks=config["model"]["n_blocks"],
    time_emb_dim=config["model"]["time_emb_dim"],
    padding_mode=config["model"]["padding_mode"],
    bilinear=config["model"]["bilinear"]
)
model = model.to(config["training"]["used_gpu"])
criterion_smooth = nn.SmoothL1Loss(reduction='mean')
criterion_mse = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
# create a lr scheduler with linear decay between the initial lr and 1e-1*lr
scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=config["training"]["num_epochs"])

# Training loop
num_epochs = config["training"]["num_epochs"]
steps_per_epoch = config["training"]["steps_per_epoch"]
config["training"]["best_loss"] = float('inf')
# Training loop
print(f"Starting training for {num_epochs} epochs with {steps_per_epoch} steps per epoch.")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (X_batch, Y_batch, T_batch) in tqdm(enumerate(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", total=steps_per_epoch):
        if i >= steps_per_epoch:
            break
        
        X_batch = X_batch.to(config["training"]["used_gpu"])
        Y_batch = Y_batch.to(config["training"]["used_gpu"])
        T_batch = T_batch.to(config["training"]["used_gpu"])
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_batch, T_batch)
        loss_smooth = criterion_smooth(outputs, Y_batch)
        loss_mse = criterion_mse(outputs, Y_batch)

        # Backward pass
        loss_smooth.backward()
        optimizer.step()

        running_loss += loss_smooth.item()

        # Log loss to wandb
        wandb.log({"batch_loss_mse": loss_smooth.item(),
                   "batch_loss_smooth": loss_mse.item(),
                   "step": epoch * steps_per_epoch + i})

    avg_loss = running_loss/steps_per_epoch
    # Update the learning rate
    scheduler.step()
    # save the model every x epochs and if the loss is lower than the previous one
    if (epoch + 1) % config["training"]["save_model_every"] == 0 or avg_loss < config["training"]["best_loss"]:
        torch.save(model.state_dict(), os.path.join(config['training']['save_path'], f'deepvel_epoch_{epoch+1}.pth'))
        print(f"Model saved at epoch {epoch+1}")
        config["training"]["best_loss"] = avg_loss

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    wandb.log({"epoch": epoch + 1, "avg_loss": avg_loss,
               "lr": optimizer.param_groups[0]['lr'], "best_loss": config["training"]["best_loss"]})

# Save the model
torch.save(model.state_dict(), os.path.join(config['training']['save_path'], f'deepvel_finished.pth'))

# Finish wandb run
wandb.finish()

