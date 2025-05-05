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
    # save the config file in the save path
    with open(os.path.join(config['training']['save_path'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

# if a previous checkpoint was provided in the configuration, load the state dict to continue the training:
model_checkpoint = None
if os.path.isfile(config["training"]["load_from"]):
    print(f"Loading model from {config['training']['load_from']}")
    model_checkpoint = torch.load(config["training"]["load_from"])

# Initialize wandb
# Initialize wandb
wandb.init(project="Deepvel_unet",
    config=config,
    # name with the basic config + current time and date
    name=f"deepvel_unet2_imsize_{config['dataset']['patch_size']}_aug_{config['dataset']['aug']}_lr_{config['training']['learning_rate']}_t_{time.strftime('%Y%m%d-%H%M%S')}",
)

# Load data from files (assumes the data folder exists and muram functions work)
test_samples = config["dataset"]["test_samples"]
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

# Create the datasets and dataloaders
dataset_train = MURAMVelocityDataset(Intensities[test_samples:], velocities[test_samples:], times[test_samples:],
                                img_size=config["dataset"]["patch_size"],
                                batch_size=config["dataset"]["batch_size"],
                                aug=config["dataset"]["aug"],
                                seed=config["dataset"]["seed"],)

dataset_test = MURAMVelocityDataset(Intensities[:test_samples], velocities[:test_samples], times[:test_samples],
                                img_size=config["dataset"]["patch_size"],
                                batch_size=config["dataset"]["batch_size"],
                                aug=config["dataset"]["aug"],
                                seed=config["dataset"]["seed"],)

dataloader_train = DataLoader(dataset_train,
                              batch_size=None,
                              num_workers=config["dataset"]["num_workers"])

dataloader_test = DataLoader(dataset_test,
                             batch_size=None,
                             num_workers=4)

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
    bilinear=config["model"]["bilinear"],
    dropout=config["model"]["dropout"],
)

if model_checkpoint is not None:
    model.load_state_dict(model_checkpoint)

model = model.to(config["training"]["used_gpu"])

criterion_mse = nn.MSELoss()
criterion_cossim = nn.CosineSimilarity(dim=1)
criterion_mae = nn.L1Loss(reduction='mean')

optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
# create a lr scheduler with linear decay between the initial lr and 1e-1*lr
# scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.05, total_iters=config["training"]["num_epochs"])
# create a lr scheduler with the best loss for a Unet with attention layers
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, min_lr = 1e-6, cooldown=50,
#                                                  patience=25, min_lr=1e-7, threshold=1e-3)
# Create a LR scheduler with the CosineAnnealingWarmRestarts (used in atention layers)
for param_group in optimizer.param_groups:
    param_group['initial_lr'] = config["training"]["learning_rate"]
    param_group['lr'] = config["training"]["learning_rate"]
# Create a cosine annealing scheduler with warm restarts
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2, eta_min=1e-7)


# Training loop
num_epochs = config["training"]["num_epochs"]
steps_per_epoch = config["training"]["steps_per_epoch"]
tests_per_epoch = config["training"]["tests_per_epoch"]
config["training"]["best_loss"] = float('inf')
# Training loop
print(f"Starting training for {num_epochs} epochs with {steps_per_epoch} steps per epoch.")
for epoch in range(num_epochs):
    model.train()
    # Reset the loss
    # Training loss
    training_loss = 0.0
    training_loss_mse = 0.0
    training_loss_cossim = 0.0
    training_loss_mae = 0.0
    # Test loss
    tests_loss = 0.0
    tests_loss_mse = 0.0
    tests_loss_cossim = 0.0
    tests_loss_mae = 0.0
    # Training loop
    for i, (X_batch, Y_batch, T_batch) in tqdm(enumerate(dataloader_train), desc=f"Epoch {epoch+1}/{num_epochs}", total=steps_per_epoch):
        if i >= steps_per_epoch:
            break
        
        X_batch = X_batch.to(config["training"]["used_gpu"])
        Y_batch = Y_batch.to(config["training"]["used_gpu"])
        T_batch = T_batch.to(config["training"]["used_gpu"])
        
        # Forward pass
        optimizer.zero_grad()
        prediction = model(X_batch, T_batch)

        # Compute the mse general loss
        loss_mse = criterion_mse(prediction, Y_batch)
        # Compute the cosine similarity loss (convert to a proper loss)
        cos_sim = criterion_cossim(prediction, Y_batch)  # Shape: [batch_size, height, width]
        loss_cossim = (1 - cos_sim).mean()  # Convert to loss and take mean to get a scalar
        # compute the mae loss over the norm of the velocities
        loss_mae = criterion_mae(torch.norm(prediction, dim=1), torch.norm(Y_batch, dim=1))
        # compute the global loss
        loss = loss_mse + 0.5*loss_cossim + loss_mae

        # Backward pass
        loss.backward()
        optimizer.step()

        # save the losses in to do the mean of the epoch
        training_loss_mse += loss_mse.item()
        training_loss_cossim += loss_cossim.item()
        training_loss_mae += loss_mae.item()
        # save the global loss
        training_loss += loss.item()

    # Test the model for a few sample each ending of the epoch
    model.eval()
    with torch.no_grad():
        for j, (X_batch, Y_batch, T_batch) in tqdm(enumerate(dataloader_test), desc=f"Test for epoch {epoch+1}/{num_epochs}", total=tests_per_epoch):
            if j >= tests_per_epoch:
                break
            X_batch = X_batch.to(config["training"]["used_gpu"])
            Y_batch = Y_batch.to(config["training"]["used_gpu"])
            T_batch = T_batch.to(config["training"]["used_gpu"])

            # Forward pass
            prediction = model(X_batch, T_batch)
            # Compute the mse general loss
            loss_mse = criterion_mse(prediction, Y_batch)
            # Compute the cosine similarity loss (convert to a proper loss)
            cos_sim = criterion_cossim(prediction, Y_batch)  # Shape: [batch_size, height, width]
            loss_cossim = (1 - cos_sim).mean()  # Convert to loss and take mean to get a scalar
            # compute the mae loss over the norm of the velocities
            loss_mae = criterion_mae(torch.norm(prediction, dim=1), torch.norm(Y_batch, dim=1))
            # compute the global loss
            loss = loss_mse + 0.5*loss_cossim + loss_mae


            # save the losses in to do the mean of the epoch
            tests_loss_mse += loss_mse.item()
            tests_loss_cossim += loss_cossim.item()
            tests_loss_mae += loss_mae.item()
            # save the global loss
            tests_loss += loss.item()

    # Compute the average loss over the epoch
    avg_training_loss = training_loss/steps_per_epoch
    avg_training_loss_mse = training_loss_mse/steps_per_epoch
    avg_training_loss_cossim = training_loss_cossim/steps_per_epoch
    avg_training_loss_mae = training_loss_mae/steps_per_epoch

    avg_testing_loss = tests_loss/tests_per_epoch
    avg_testing_loss_mse = tests_loss_mse/tests_per_epoch
    avg_testing_loss_cossim = tests_loss_cossim/tests_per_epoch
    avg_testing_loss_mae = tests_loss_mae/tests_per_epoch


    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_training_loss:.4f}, Testing Loss: {avg_testing_loss:.4f}")
    wandb.log({"epoch": epoch + 1, "avg_loss": avg_training_loss,   "test_loss": avg_testing_loss,
               "avg_train_loss_mse": avg_training_loss_mse,         "avg_test_loss_mse": avg_testing_loss_mse,
               "avg_train_loss_cossim": avg_training_loss_cossim,   "avg_test_loss_cossim": avg_testing_loss_cossim,
               "avg_train_loss_mae": avg_training_loss_mae,         "avg_test_loss_mae": avg_testing_loss_mae,
               "lr": optimizer.param_groups[0]['lr'],               "best_loss": config["training"]["best_loss"]})

    # Update the learning rate
    scheduler.step()
    # save the model every x epochs and if the loss is lower than the previous one
    if (epoch + 1) % config["training"]["save_model_every"] == 0:
        # Save checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,  # Optional: save epoch for tracking
        }
        torch.save(checkpoint, os.path.join(config['training']['save_path'], f'deepvel_epoch_{epoch+1}.pth'))
        print(f"Model saved for checkpoint at epoch {epoch+1}")

    if avg_testing_loss < config["training"]["best_loss"]:
        # Save checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,  # Optional: save epoch for tracking
        }
        torch.save(checkpoint, os.path.join(config['training']['save_path'], f'deepvel_best_{epoch+1}.pth'))
        config["training"]["best_loss"] = avg_testing_loss
        print(f"Best model saved at epoch {epoch+1} with loss {avg_testing_loss:.4f}")

# Save the model
torch.save(model.state_dict(), os.path.join(config['training']['save_path'], f'deepvel_finished.pth'))

# Finish wandb run
wandb.finish()
