"""
CosmicDreams: Dream Generation Script

Load a trained ResNet-Rollout model and generate "dreams" (predictions) by
auto-regressively predicting future latent states.

The model starts with 4 initial context frames and iteratively predicts the
next frame, feeding predictions back as input.

Usage:
    python src/dream.py --checkpoint ../checkpoints/resnet_rollout_best.pt
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse

# Import local modules
from modules import ResidualBlock3D
from dataset import LatentDynamicsDataset


# ============================================================================
# ARCHITECTURE: ResNet-Rollout (must match training)
# ============================================================================

class LatentDynamicsPredictor(nn.Module):
    """ResNet-Rollout architecture (same as train.py)"""
    
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 16,
        hidden_channels: int = 64,
        num_res_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        
        # Input projection
        self.conv_in = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.norm_in = nn.GroupNorm(8, hidden_channels)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock3D(hidden_channels, dropout)
            for _ in range(num_res_blocks)
        ])
        
        # Output projection
        self.conv_out = nn.Conv3d(hidden_channels, out_channels, kernel_size=3, padding=1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C*context_window, 1, H, W]
        Returns:
            z_next: [B, C, 1, H, W]
        """
        x = self.conv_in(x)
        x = self.norm_in(x)
        x = F.silu(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        z_next = self.conv_out(x)
        
        return z_next


# ============================================================================
# DREAM: Auto-regressive Prediction
# ============================================================================

def dream_resnet_rollout(
    model: nn.Module,
    initial_context: torch.Tensor,
    num_frames: int,
    device: torch.device,
    dtype: torch.dtype,
    latent_mean: float,
    latent_std: float,
    context_window: int = 4,
):
    """
    Generate dreams by auto-regressively predicting future latent states.
    
    Args:
        model: Trained ResNet-Rollout model
        initial_context: Initial context frames [C*context_window, 1, H, W]
        num_frames: Number of frames to predict
        device: Device to run on
        dtype: Data type (bfloat16 or float32)
        latent_mean: Mean for normalization
        latent_std: Std for normalization
        context_window: Number of context frames (default: 4)
    
    Returns:
        predicted_latents: Predicted latent frames [C, num_frames, H, W]
    """
    model.eval()
    
    C = initial_context.shape[0] // context_window
    
    # Stack initial context
    context_frames = []
    for i in range(context_window):
        frame = initial_context[i*C:(i+1)*C, 0:1, :, :]
        context_frames.append(frame)
    
    current_context = torch.cat(context_frames, dim=0).unsqueeze(0).to(device, dtype=dtype)
    current_context_norm = (current_context - latent_mean) / (latent_std + 1e-8)
    
    predicted_frames = []
    
    print(f"🎯 Dreaming with ResNet-Rollout...")
    with torch.no_grad():
        for i in tqdm(range(num_frames), desc="Dreaming", ncols=100):
            # Predict next frame
            z_next_norm = model(current_context_norm)
            
            # Denormalize
            z_next_denorm = z_next_norm * latent_std + latent_mean
            
            predicted_frames.append(z_next_denorm.squeeze(0))
            
            # Update context
            current_context_norm = torch.cat([
                current_context_norm[:, 16:, :, :, :],
                z_next_norm
            ], dim=1)
    
    predicted_latents = torch.cat(predicted_frames, dim=1)
    
    return predicted_latents


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate dreams with ResNet-Rollout")
    parser.add_argument("--checkpoint", type=str, default="../checkpoints/resnet_rollout_best.pt",
                       help="Path to trained model checkpoint")
    parser.add_argument("--latent_path", type=str, default="../latent.pt",
                       help="Path to latent.pt file (for initial context)")
    parser.add_argument("--stats_path", type=str, default="../latent_stats.pt",
                       help="Path to normalization statistics")
    parser.add_argument("--output_path", type=str, default="../results/dreamed_latents.pt",
                       help="Path to save dreamed latents")
    parser.add_argument("--context_window", type=int, default=4,
                       help="Number of context frames")
    parser.add_argument("--hidden_channels", type=int, default=64,
                       help="Hidden channels in model")
    parser.add_argument("--num_res_blocks", type=int, default=4,
                       help="Number of residual blocks")
    
    args = parser.parse_args()
    
    # GPU setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    
    print("="*80)
    print("GENERATING RESNET-ROLLOUT DREAM")
    print("="*80)
    print(f"🚀 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"📦 Checkpoint: {args.checkpoint}")
    print("="*80)
    
    # Load dataset for initial context and stats
    print("\n📊 Loading dataset...")
    dataset = LatentDynamicsDataset(
        latent_path=args.latent_path,
        context_window=args.context_window,
        rollout_steps=4,
        stats_path=args.stats_path
    )
    
    latent_mean = dataset.latent_mean
    latent_std = dataset.latent_std
    
    print(f"✓ Loaded: latent shape {dataset.latent.shape}")
    print(f"✓ Normalization: mean={latent_mean:.6f}, std={latent_std:.6f}")
    
    # Load model
    print("\n🏗️  Loading model...")
    model = LatentDynamicsPredictor(
        in_channels=args.context_window * 16,
        out_channels=16,
        hidden_channels=args.hidden_channels,
        num_res_blocks=args.num_res_blocks,
        dropout=0.1
    ).to(device, dtype=dtype)
    
    # Load checkpoint
    print(f"📥 Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # Handle torch.compile wrapper
    state_dict = ckpt['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    print(f"✓ Loaded checkpoint from epoch {ckpt['epoch']}, val loss: {ckpt['val_loss']:.4f}")
    
    # Get initial context
    initial_context = dataset.latent[:, :args.context_window, :, :]
    C, T, H, W = initial_context.shape
    initial_context_stacked = initial_context.reshape(C * T, 1, H, W)
    
    # Dream
    num_dream_frames = dataset.latent.shape[1] - args.context_window
    print(f"\n🎯 Generating {num_dream_frames} dream frames...")
    
    predicted_latents = dream_resnet_rollout(
        model, initial_context_stacked, num_dream_frames,
        device, dtype, latent_mean, latent_std,
        args.context_window
    )
    
    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(predicted_latents.cpu(), args.output_path)
    
    print(f"\n✓ Saved: {args.output_path}")
    print(f"  Latents shape: {predicted_latents.shape}")
    print("="*80)
    print("🎉 SUCCESS: Dream generation complete!")
    print("\nNext steps:")
    print("  1. Decode latents to images using Cosmos decoder")
    print("  2. Analyze tracking with analyze.py")
    print("="*80)


if __name__ == "__main__":
    main()

