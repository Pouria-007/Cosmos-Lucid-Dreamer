# 🌌 Cosmos-Lucid-Dreamer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Read%20Technical%20Report-brightgreen.svg)](docs/TECHNICAL_REPORT.md)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

> **"Physics is observable only when you look at history."**

**ResNet-Rollout**: A simple, fast, and highly accurate latent world model for video prediction.

**Performance**: 71.08 px RMSE | 99% Detection | ~1M Parameters | 15 Minutes Training

---

## 🏆 Key Results

We achieved a **56% reduction in error** compared to high-capacity baselines by using a simple ResNet with a 4-frame context window.

| Metric | ResNet-Rollout (Ours) | Previous Best | Improvement |
|--------|-----------------------|---------------|-------------|
| **RMSE** | **71.08 pixels** | 165.20 px | **56% Better** |
| **Detection Rate** | **99%** | 100% | **Parity** |
| **Model Size** | **~1.02 M** | ~84 M | **84× Smaller** |
| **Training Time** | **15 mins** | 45 mins | **3× Faster** |

---

## 🔬 The Science: From "Fever Dream" to "Lucid Dream"

**ResNet-Rollout** is a latent world model that predicts future states in the [Cosmos tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer) latent space. It combines three key innovations:

1. **Simple ResNet Architecture** (~1M params, 4 residual blocks, no skip connections)
2. **4-Frame Context Window** (provides velocity information, eliminates blur)
3. **Teacher-Less Rollout Training** (enforces temporal consistency)

### Why It Works: The "Lucid Dream" Principle

Single-frame models suffer from **Posterior Collapse**—without velocity information, they must average over all possible futures, resulting in blur. A 4-frame context window provides observable velocity, disambiguating future predictions and enabling sharp, deterministic output.

![Lucid Dream Concept](docs/figure_1_concept.png)

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Pouria-007/Cosmos-Lucid-Dreamer.git
cd Cosmos-Lucid-Dreamer

# Install dependencies
pip install -r requirements.txt
```

### 2. Prerequisites

You need:
- **Tokenized latent data** (`latent.pt`): Use [Cosmos-Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer) to encode your video frames
- **Normalization statistics** (`latent_stats.pt`): Mean and std of latent space
- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 5090)

### 3. Training (15 Minutes)

```bash
cd src
python train.py --epochs 500 --batch_size 32 --lr 1e-4 \
  --context_window 4 \
  --rollout_steps 4 \
  --output_dir ../checkpoints
```

### 4. Dreaming (Inference)

```bash
cd src
python dream.py --checkpoint ../checkpoints/resnet_rollout_best.pt \
  --output_path ../results/dreamed_latents.pt
```

### 5. Analysis

```bash
cd src
python analyze.py --ground_truth ../cosmic_data/frames \
  --dreamed ../results/decoded_frames \
  --output_figure ../results/tracking_analysis.png
```

---

## 📊 Architecture

We use a surprisingly simple architecture to prove that **data structure matters more than model depth**.

### Model Specification

```python
Input:  [Batch, 64, 1, 64, 64]  # Stack of 4 Latent Frames (16ch each)
Output: [Batch, 16, 1, 64, 64]  # Next Latent Frame

Architecture:
  - Conv3d(64 → 64) + GroupNorm + SiLU
  - 4× ResidualBlock3D(64)
  - Conv3d(64 → 16)

Parameters: ~1,024,720 (~1M)
```

![Architecture](docs/figure_2_architecture.png)

---

## 📈 Performance Comparison

The "Inverted-U" curve shows that **over-parameterization actually hurts performance** in autoregressive tasks due to stability issues.

| Model Strategy | RMSE (px) | Detection | Status |
|----------------|-----------|-----------|--------|
| **ResNet-Rollout (Ours)** | **71.08** | **99%** | **✅ Winner** |
| Probabilistic Plus | 77.81 | 100% | ⚠️ Jittery |
| Sharp-Shooter (GDL) | 108.0 | 59% | ❌ Unstable |
| Rollout U-Net | 165.2 | 100% | ❌ Blurry |
| Hero Model (394M) | N/A | 30% | ❌ Failed |

![Results](docs/figure_3_results.png)

---

## 🎬 Demo Video

**Left**: Ground Truth | **Center**: Probabilistic (Jitter) | **Right**: ResNet-Rollout (Sharp)

[Watch Comparison Video](docs/demo_video.mp4)

---

## 📁 Repository Structure

```
Cosmos-Lucid-Dreamer/
├── src/
│   ├── train.py          # Training script (Teacher-less Rollout)
│   ├── dream.py          # Inference script
│   ├── analyze.py        # RMSE and Centroid tracking analysis
│   ├── modules.py        # ResNet Architecture
│   └── dataset.py        # Context-Aware Dataloader
├── scripts/
│   └── generate_paper_figures.py  # Reproduction for paper figures
├── checkpoints/
│   └── resnet_rollout_best.pt     # Trained model (71.08 px RMSE)
├── docs/
│   ├── TECHNICAL_REPORT.md        # Detailed technical report
│   ├── figure_1_concept.png       # Concept diagram
│   ├── figure_2_architecture.png  # Architecture diagram
│   ├── figure_3_results.png       # Performance comparison
│   └── demo_video.mp4             # Comparison video
├── results/                       # Output directory
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md                      # This file
```

---

## 🔬 Key Insights

### 1. Small Models Beat Large Models

- **1M parameters** outperforms **84M parameters**
- Less overfitting → better generalization
- Faster training, easier deployment

### 2. Context > Architecture

- **4-frame context window** provides velocity information
- More important than skip connections (U-Net)
- Disambiguates future predictions → eliminates blur

### 3. Motion-Weighted Loss is Essential

- **10× weight on moving pixels**
- Forces model to track objects (not background)
- **99% detection rate**

### 4. Teacher-Less Rollout Enforces Consistency

- Predictions feed back as input
- Prevents error accumulation
- Smooth, stable trajectories

---

## 📝 Citation

If you use this code or the "Lucid Dream" concept in your research, please cite:

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

**Publication**: [TechRxiv Article](https://www.techrxiv.org/users/1001132/articles/1373525-cosmic-dreams-lucid-dreaming-in-latent-space-via-temporal-context-windows)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [NVIDIA Cosmos Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer) for the latent space encoder/decoder
- [World Models (Ha & Schmidhuber, 2018)](https://worldmodels.github.io/) for inspiration
- The PyTorch team for excellent tools

---

**Status**: ✅ Production Ready | 📦 Tested on NVIDIA RTX 5090
