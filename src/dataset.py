"""
CosmicDreams: Dataset Module

Handles loading and preprocessing of latent tensors for training/inference.
"""

import os
import torch
from torch.utils.data import Dataset


class LatentDynamicsDataset(Dataset):
    """
    Dataset for training latent dynamics with temporal context window.
    
    Given a sequence of latent frames, this dataset creates training samples
    where the input is `context_window` consecutive frames and the target
    is the next `rollout_steps` frames.
    
    Args:
        latent_path: Path to latent.pt file with shape [C, T, H, W] or [B, C, T, H, W]
        context_window: Number of past frames to use as input (default: 4)
        rollout_steps: Number of future frames to predict (default: 4)
        stats_path: Path to latent statistics (mean, std) for normalization
        overfit_mode: If True, only use first training sample (for debugging)
    
    Returns:
        z_context: Stacked input frames [C*context_window, 1, H, W]
        z_target: Target frames [C, rollout_steps, H, W]
    """
    
    def __init__(
        self,
        latent_path: str = "latent.pt",
        context_window: int = 4,
        rollout_steps: int = 4,
        stats_path: str = "latent_stats.pt",
        overfit_mode: bool = False
    ):
        # Load latent tensor
        self.latent = torch.load(latent_path)
        
        # Remove batch dimension if present: [B, C, T, H, W] -> [C, T, H, W]
        if self.latent.dim() == 5:
            self.latent = self.latent.squeeze(0)
        
        self.context_window = context_window
        self.rollout_steps = rollout_steps
        self.overfit_mode = overfit_mode
        
        # Load normalization statistics
        if os.path.exists(stats_path):
            stats = torch.load(stats_path)
            self.latent_mean = stats["mean"]
            self.latent_std = stats["std"]
            print(f"📊 Loaded normalization stats - mean: {self.latent_mean:.6f}, std: {self.latent_std:.6f}")
        else:
            # Calculate from data if stats file doesn't exist
            self.latent_mean = self.latent.float().mean().item()
            self.latent_std = self.latent.float().std().item()
            print(f"📊 Calculated normalization stats - mean: {self.latent_mean:.6f}, std: {self.latent_std:.6f}")
        
        # latent shape: [C, T, H, W] = [16, T, 64, 64]
        self.num_timesteps = self.latent.shape[1]
        
        # Need context_window + rollout_steps frames for each sample
        self.num_samples = max(1, self.num_timesteps - context_window - rollout_steps + 1)
        
        print(f"📊 Dataset Configuration:")
        print(f"   Latent shape: {self.latent.shape}")
        print(f"   Context window: {context_window} frames")
        print(f"   Rollout steps: {rollout_steps} frames")
        print(f"   Training samples: {self.num_samples}")
        if overfit_mode:
            print(f"   ⚠️  OVERFIT MODE - using only first sample")
    
    def __len__(self):
        """Return number of training samples"""
        if self.overfit_mode:
            return 1
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a training sample.
        
        Args:
            idx: Sample index
        
        Returns:
            z_context: Input frames [C*context_window, 1, H, W]
            z_target: Target frames [C, rollout_steps, H, W]
        """
        if self.overfit_mode:
            idx = 0
        
        # Input: context_window consecutive frames stacked along channel dimension
        context_frames = []
        for i in range(self.context_window):
            frame = self.latent[:, idx + i:idx + i + 1, :, :]  # [C, 1, H, W]
            context_frames.append(frame)
        
        z_context = torch.cat(context_frames, dim=0)  # [C*context_window, 1, H, W]
        
        # Target: next rollout_steps frames
        target_start = idx + self.context_window
        z_target = self.latent[:, target_start:target_start + self.rollout_steps, :, :]  # [C, rollout_steps, H, W]
        
        return z_context, z_target

