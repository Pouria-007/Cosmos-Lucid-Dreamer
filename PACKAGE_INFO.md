# 📦 Cosmos-Lucid-Dreamer

## Contents

### Core Code (`src/`)
- `train.py` - Training script with ResNet-Rollout
- `dream.py` - Inference/dream generation script
- `analyze.py` - Tracking analysis tool
- `modules.py` - ResidualBlock3D building block
- `dataset.py` - LatentDynamicsDataset class

### Model (`checkpoints/`)
- `resnet_rollout_best.pt` - Trained model (5.9 MB, 71.08 px RMSE)

### Documentation (`docs/`)
- `TECHNICAL_REPORT.md` - Full technical report
- `figure_1_concept.png` - The "Lucid Dream" concept diagram
- `figure_2_architecture.png` - ResNet-Rollout architecture
- `figure_3_results.png` - Performance comparison
- `demo_video.mp4` - Side-by-side comparison video (668 KB)

### Scripts (`scripts/`)
- `generate_paper_figures.py` - Generate publication figures

### Configuration
- `README.md` - Main documentation
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules

---

## 🚀 Quick Test

```bash
cd src

# Test imports
python -c "from modules import ResidualBlock3D; print('✓ modules.py works')"
python -c "from dataset import LatentDynamicsDataset; print('✓ dataset.py works')"

# Check model file
ls -lh ../checkpoints/resnet_rollout_best.pt

# View documentation
cat ../README.md | head -20
```

---

## 📋 Usage Instructions

### 1. Prepare Your Data

You need to tokenize your video frames using [NVIDIA Cosmos Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer):

```python
# Example tokenization (not included in this package)
import torch
from cosmos_tokenizer import Encoder

encoder = torch.jit.load("encoder.jit")
frames = load_your_frames()  # [1, 3, T, H, W]
latent = encoder(frames)  # [1, 16, T//8, H//8, W//8]
torch.save(latent, "latent.pt")

# Calculate normalization stats
latent_mean = latent.mean().item()
latent_std = latent.std().item()
torch.save({"mean": latent_mean, "std": latent_std}, "latent_stats.pt")
```

### 2. Train the Model

```bash
cd src
python train.py --epochs 500 --batch_size 32 --lr 1e-4 \
  --latent_path /path/to/latent.pt \
  --stats_path /path/to/latent_stats.pt \
  --output_dir ../checkpoints
```

### 3. Generate Dreams

```bash
python dream.py --checkpoint ../checkpoints/resnet_rollout_best.pt \
  --latent_path /path/to/latent.pt \
  --stats_path /path/to/latent_stats.pt \
  --output_path ../results/dreamed_latents.pt
```

### 4. Decode and Analyze

```python
# Decode latents using Cosmos decoder (not included)
decoder = torch.jit.load("decoder.jit")
dreamed_latents = torch.load("../results/dreamed_latents.pt")
decoded_frames = decoder(dreamed_latents)

# Analyze tracking
python analyze.py --ground_truth /path/to/gt_frames \
  --dreamed /path/to/decoded_frames \
  --output_figure ../results/tracking_analysis.png
```

---

## 🔧 Dependencies

### Required
- Python 3.10+
- PyTorch 2.0+ (with CUDA support recommended)
- NumPy, Pillow, tqdm

### Optional
- Matplotlib (for visualization)
- OpenCV (for video processing)

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 📊 Expected Performance

With the included checkpoint (`resnet_rollout_best.pt`):
- **RMSE**: 71.08 pixels
- **Detection Rate**: 99% (96/97 frames)
- **Inference Speed**: ~0.015s per frame on RTX 5090

---

## 🎯 Key Features

1. **Small & Fast**: Only ~1M parameters, trains in 15 minutes
2. **Accurate**: 71.08 px RMSE (8.6% better than previous best)
3. **Stable**: 99% detection rate, no flicker or zig-zag
4. **Deterministic**: Reproducible predictions (no stochastic sampling)
5. **Simple**: Pure ResNet, no complex skip connections

---

## 🐛 Troubleshooting

### ImportError: No module named 'modules'

Make sure you're in the `src/` directory:
```bash
cd src
python train.py --help
```

### CUDA out of memory

Reduce batch size:
```bash
python train.py --batch_size 16  # or 8
```

### Model checkpoint not found

Check the checkpoint path:
```bash
ls -lh ../checkpoints/resnet_rollout_best.pt
```

---

## 📝 Citation

```bibtex
@article{cosmos_lucid_dreamer_2025,
  title={Cosmic Dreams: Lucid Dreaming in Latent Space via Temporal Context Windows},
  author={Javaheri, Pouria},
  journal={TechRxiv},
  year={2025},
  doi={10.36227/techrxiv.176784313.32092392/v1},
  url={https://www.techrxiv.org/users/1001132/articles/1373525}
}
```

---

## 🎓 Research Paper

For detailed technical analysis, see:
- `docs/TECHNICAL_REPORT.md` - Full technical report
- `docs/figure_1_concept.png` - Concept explanation
- `docs/figure_2_architecture.png` - Architecture details
- `docs/figure_3_results.png` - Performance comparison

---

## Repository Structure

```
Cosmos-Lucid-Dreamer/
├── src/           (5 Python files)
├── checkpoints/   (trained model)
├── docs/          (figures and technical report)
├── scripts/       (figure generation)
├── results/       (for outputs)
└── config files   (README, requirements, .gitignore)
```

---

**No Cosmos Tokenizer files included** - users must download separately from NVIDIA.

