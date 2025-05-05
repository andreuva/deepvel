import numpy as np
import torch
from torch.utils.data import IterableDataset

def data_generator(Intensities, velocities, times, batch_size=32, img_size=256, aug=None, seed=None):
    """
    Generator that yields batches of augmented samples.
    """
    np.random.seed(seed)
    n_imgs = len(Intensities)
    if n_imgs < 3:
        raise ValueError("Need at least 3 frames to generate a triplet.")

    while True:
        X_batch = []
        Y_batch = []
        T_batch = []
        for _ in range(batch_size):
            idx = np.random.randint(1, n_imgs - 1)
            imgs = [Intensities[idx - 1], Intensities[idx], Intensities[idx + 1]]
            times_triplet = [times[idx - 1], times[idx], times[idx + 1]]
            vx_full, vy_full = velocities[idx]

            H, W = imgs[0].shape
            if H < img_size or W < img_size:
                raise ValueError(f"Intensity images must be at least ({img_size}, {img_size}).")
            i_start = np.random.randint(0, H - img_size + 1)
            j_start = np.random.randint(0, W - img_size + 1)
            region = (slice(i_start, i_start + img_size), slice(j_start, j_start + img_size))

            imgs_cropped = [img[region] for img in imgs]
            vx_crop = vx_full[region]
            vy_crop = vy_full[region]
            Y_sample = np.stack([vx_crop, vy_crop], axis=-1)

            # Data augmentation
            if aug in ["rotation", "all"]:
                k = np.random.choice([0, 1, 2, 3])  # 0, 90, 180, 270 degrees
                imgs_cropped = [np.rot90(img, k) for img in imgs_cropped]
                Y_sample = np.rot90(Y_sample, k)
                # Adjust velocity components for rotation
                if k == 1:  # 90 degrees
                    Y_sample = np.stack([-Y_sample[..., 1], Y_sample[..., 0]], axis=-1)
                elif k == 2:  # 180 degrees
                    Y_sample = np.stack([-Y_sample[..., 0], -Y_sample[..., 1]], axis=-1)
                elif k == 3:  # 270 degrees
                    Y_sample = np.stack([Y_sample[..., 1], -Y_sample[..., 0]], axis=-1)

            if aug in ["flip", "all"]:
                if np.random.rand() > 0.5:  # Horizontal flip
                    imgs_cropped = [np.fliplr(img) for img in imgs_cropped]
                    Y_sample = np.stack([-Y_sample[..., 0], Y_sample[..., 1]], axis=-1)
                if np.random.rand() > 0.5:  # Vertical flip
                    imgs_cropped = [np.flipud(img) for img in imgs_cropped]
                    Y_sample = np.stack([Y_sample[..., 0], -Y_sample[..., 1]], axis=-1)

            X_sample = np.stack(imgs_cropped, axis=-1)
            X_batch.append(X_sample)
            Y_batch.append(Y_sample)
            T_batch.append(times_triplet)

        X_batch = np.array(X_batch)  # (batch_size, img_size, img_size, 3)
        Y_batch = np.array(Y_batch)  # (batch_size, img_size, img_size, 2)
        T_batch = np.array(T_batch)  # (batch_size, 3)

        yield X_batch, Y_batch, T_batch


################################################################################
# Dataset Class: MURAMVelocityDataset
################################################################################
# Data generator as an IterableDataset
class MURAMVelocityDataset(IterableDataset):
    def __init__(self, Intensities, velocities, times, num_frames=3, batch_size=32, img_size=256, aug=None, seed=None):
        self.Intensities = np.array(Intensities)
        # Normalize the intensities
        self.Intensities = (self.Intensities - self.Intensities.mean())/(np.std(self.Intensities)*2)
        self.velocities = np.array(velocities)/1e6
        self.times = np.array(times)/1e1
        self.batch_size = batch_size
        self.img_size = img_size
        self.aug = aug
        self.seed = seed
        self.num_frames = num_frames
        if self.num_frames != 2 and self.num_frames != 3:
            raise ValueError("num_frames must be 2 or 3.")
        self.n_imgs = len(Intensities)
        if self.n_imgs < self.num_frames:
            raise ValueError(f"Need at least {self.num_frames} frames to generate a sequence.")
        if seed is not None:
            np.random.seed(seed)

    def __iter__(self):
        while True:
            X_batch = []
            Y_batch = []
            T_batch = []
            for _ in range(self.batch_size):
                # Select the index of the middle image in the sequence (where the velocity is extracted)
                idx = np.random.randint(1, self.n_imgs - 1)
                # Select num_frames consecutive images for the sequence
                
                if self.num_frames == 3:
                    imgs = [self.Intensities[idx-1], self.Intensities[idx], self.Intensities[idx+1]]
                    times_triplet = [self.times[idx-1]-self.times[idx], self.times[idx], self.times[idx+1]-self.times[idx]]
                elif self.num_frames == 2:
                    imgs = [self.Intensities[idx-1], self.Intensities[idx]]
                    times_triplet = [self.times[idx-1]-self.times[idx], self.times[idx]]
                # Select the velocity field corresponding to the middle image
                vx_full, vy_full = self.velocities[idx]

                H, W = imgs[0].shape
                if H < self.img_size or W < self.img_size:
                    raise ValueError(f"Intensity images must be at least ({self.img_size}, {self.img_size}).")
                i_start = np.random.randint(0, H - self.img_size + 1)
                j_start = np.random.randint(0, W - self.img_size + 1)
                region = (slice(i_start, i_start + self.img_size), slice(j_start, j_start + self.img_size))

                imgs_cropped = [img[region] for img in imgs]
                vx_crop = vx_full[region]
                vy_crop = vy_full[region]
                Y_sample = np.stack([vx_crop, vy_crop], axis=-1)

                # Data augmentation
                if self.aug in ["rotation", "all"]:
                    k = np.random.choice([0, 1, 2, 3])
                    imgs_cropped = [np.rot90(img, k) for img in imgs_cropped]
                    Y_sample = np.rot90(Y_sample, k)
                    if k == 1:
                        Y_sample = np.stack([-Y_sample[..., 1], Y_sample[..., 0]], axis=-1)
                    elif k == 2:
                        Y_sample = np.stack([-Y_sample[..., 0], -Y_sample[..., 1]], axis=-1)
                    elif k == 3:
                        Y_sample = np.stack([Y_sample[..., 1], -Y_sample[..., 0]], axis=-1)

                if self.aug in ["flip", "all"]:
                    if np.random.rand() > 0.5:
                        imgs_cropped = [np.fliplr(img) for img in imgs_cropped]
                        Y_sample = np.stack([-Y_sample[..., 0], Y_sample[..., 1]], axis=-1)
                    if np.random.rand() > 0.5:
                        imgs_cropped = [np.flipud(img) for img in imgs_cropped]
                        Y_sample = np.stack([Y_sample[..., 0], -Y_sample[..., 1]], axis=-1)

                X_sample = np.stack(imgs_cropped, axis=-1)
                X_batch.append(X_sample)
                Y_batch.append(Y_sample)
                T_batch.append(times_triplet)

            X_batch = np.array(X_batch)  # (batch_size, img_size, img_size, 3)
            Y_batch = np.array(Y_batch)  # (batch_size, img_size, img_size, 2)
            T_batch = np.array(T_batch)  # (batch_size, 3)

            # Convert to PyTorch tensors
            X_batch = torch.tensor(X_batch, dtype=torch.float32).permute(0, 3, 1, 2)  # (batch_size, 3, img_size, img_size)
            Y_batch = torch.tensor(Y_batch, dtype=torch.float32).permute(0, 3, 1, 2)  # (batch_size, 2, img_size, img_size)
            T_batch = torch.tensor(T_batch, dtype=torch.float32)  # (batch_size, 3)

            yield X_batch, Y_batch, T_batch

        