"""
Generate Publication-Quality Figures for Cosmos-Isaac World Model Paper

Figure 1: The "Lucid Dream" Concept (Why 4-Frame Context Works)
Figure 2: ResNet-Rollout Architecture Diagram

Style: CVPR/NeurIPS academic standard
Resolution: 300 DPI
Formats: PNG + PDF (for LaTeX)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Wedge
import numpy as np
from matplotlib import rcParams

# Set publication-quality defaults
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 1.5
rcParams['pdf.fonttype'] = 42  # TrueType fonts for PDF
rcParams['ps.fonttype'] = 42


# ============================================================================
# FIGURE 1: THE "LUCID DREAM" CONCEPT
# ============================================================================

def generate_figure_1_concept():
    """
    Figure 1: Why 4-Frame Context Eliminates Blur
    
    Left: Single-frame input → Posterior collapse → Blur
    Right: 4-frame context → Observable velocity → Sharp
    """
    
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # ========================================================================
    # LEFT SIDE: The Nightmare (Single Frame = Blur)
    # ========================================================================
    
    ax_left.set_xlim(0, 10)
    ax_left.set_ylim(0, 10)
    ax_left.set_aspect('equal')
    ax_left.axis('off')
    
    # Current position (single frame)
    current_x, current_y = 3, 5
    ax_left.scatter([current_x], [current_y], s=400, c='red', marker='o', 
                    edgecolors='darkred', linewidths=3, zorder=10, label='Current State (t)')
    
    # Multiple possible futures (uncertainty)
    np.random.seed(42)
    num_futures = 8
    future_angles = np.linspace(0, 2*np.pi, num_futures, endpoint=False)
    future_trajectories = []
    
    for angle in future_angles:
        # Generate curved trajectory
        t = np.linspace(0, 1, 20)
        x = current_x + 3 * t * np.cos(angle) + 0.5 * np.sin(3*t) * np.cos(angle + np.pi/2)
        y = current_y + 3 * t * np.sin(angle) + 0.5 * np.sin(3*t) * np.sin(angle + np.pi/2)
        
        ax_left.plot(x, y, 'gray', linestyle='--', alpha=0.4, linewidth=2)
        future_trajectories.append((x[-1], y[-1]))
    
    # Show "averaged" prediction (blur)
    avg_x = np.mean([p[0] for p in future_trajectories])
    avg_y = np.mean([p[1] for p in future_trajectories])
    
    # Draw blur cloud
    blur_circle = Circle((avg_x, avg_y), 1.5, color='red', alpha=0.15, zorder=5)
    ax_left.add_patch(blur_circle)
    ax_left.scatter([avg_x], [avg_y], s=400, c='red', marker='X', 
                    alpha=0.5, edgecolors='darkred', linewidths=2, zorder=6)
    
    # Labels
    ax_left.text(current_x, current_y - 1.2, r'$z_t$', fontsize=16, ha='center', 
                weight='bold', color='darkred')
    ax_left.text(avg_x, avg_y + 1.8, r'$\mathbb{E}[z_{t+1}]$', fontsize=16, ha='center',
                weight='bold', color='red', alpha=0.7)
    ax_left.text(avg_x, avg_y + 2.5, '(Blur!)', fontsize=14, ha='center',
                style='italic', color='red')
    
    # Title
    ax_left.text(5, 9.5, 'Single-Frame Input', fontsize=18, ha='center', 
                weight='bold', color='darkred')
    ax_left.text(5, 8.7, r'$P(z_{t+1} | z_t)$ = Posterior Collapse', fontsize=13, 
                ha='center', style='italic', color='gray')
    ax_left.text(5, 0.5, '❌ Multiple Possible Futures → Average → Blur', 
                fontsize=12, ha='center', color='darkred', weight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='mistyrose', edgecolor='darkred', linewidth=2))
    
    # ========================================================================
    # RIGHT SIDE: The Lucid Dream (4-Frame Context = Sharp)
    # ========================================================================
    
    ax_right.set_xlim(0, 10)
    ax_right.set_ylim(0, 10)
    ax_right.set_aspect('equal')
    ax_right.axis('off')
    
    # Historical trajectory (4 frames)
    history_x = np.array([2, 3, 4, 5])
    history_y = np.array([3, 4, 5, 6])
    
    # Draw trajectory line
    ax_right.plot(history_x, history_y, 'red', linewidth=4, alpha=0.7, zorder=5)
    
    # Draw historical points
    for i, (x, y) in enumerate(zip(history_x, history_y)):
        alpha = 0.4 + 0.2 * (i / 3)
        size = 200 + 100 * (i / 3)
        ax_right.scatter([x], [y], s=size, c='red', marker='o', 
                        edgecolors='darkred', linewidths=2, alpha=alpha, zorder=10-i)
        ax_right.text(x - 0.7, y, rf'$z_{{t-{3-i}}}$', fontsize=14, ha='center',
                     weight='bold', color='darkred', alpha=alpha)
    
    # Calculate velocity (observable from context)
    velocity_x = history_x[-1] - history_x[-2]
    velocity_y = history_y[-1] - history_y[-2]
    
    # Predicted next position (sharp!)
    next_x = history_x[-1] + velocity_x
    next_y = history_y[-1] + velocity_y
    
    # Draw prediction arrow
    arrow = FancyArrowPatch((history_x[-1], history_y[-1]), (next_x, next_y),
                           arrowstyle='->', mutation_scale=40, linewidth=4,
                           color='limegreen', zorder=20)
    ax_right.add_patch(arrow)
    
    # Draw predicted point (sharp!)
    ax_right.scatter([next_x], [next_y], s=500, c='limegreen', marker='*', 
                    edgecolors='darkgreen', linewidths=3, zorder=25)
    ax_right.text(next_x + 0.7, next_y, r'$z_{t+1}$', fontsize=16, ha='center',
                 weight='bold', color='darkgreen')
    ax_right.text(next_x, next_y - 1.0, '(Sharp!)', fontsize=14, ha='center',
                 style='italic', color='limegreen')
    
    # Velocity annotation
    mid_x = (history_x[-1] + next_x) / 2
    mid_y = (history_y[-1] + next_y) / 2 + 0.5
    ax_right.text(mid_x, mid_y, r'$\vec{v} = \frac{dz}{dt}$', fontsize=14, 
                 ha='center', style='italic', color='darkgreen',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', 
                          edgecolor='darkgreen', linewidth=1.5, alpha=0.8))
    
    # Title
    ax_right.text(5, 9.5, '4-Frame Context Window', fontsize=18, ha='center', 
                 weight='bold', color='darkgreen')
    ax_right.text(5, 8.7, r'$P(z_{t+1} | z_{t-3:t})$ = Observable Velocity', fontsize=13, 
                 ha='center', style='italic', color='gray')
    ax_right.text(5, 0.5, '✅ Single Future → Sharp Prediction', 
                 fontsize=12, ha='center', color='darkgreen', weight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', 
                          edgecolor='darkgreen', linewidth=2))
    
    # ========================================================================
    # Main Title
    # ========================================================================
    
    fig.suptitle('Why Context Window Eliminates Blur: The "Lucid Dream" Principle', 
                fontsize=20, weight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    plt.savefig('figure_1_concept.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('figure_1_concept.pdf', bbox_inches='tight', facecolor='white')
    print("✓ Saved: figure_1_concept.png + figure_1_concept.pdf")
    
    plt.close()


# ============================================================================
# FIGURE 2: RESNET-ROLLOUT ARCHITECTURE
# ============================================================================

def generate_figure_2_architecture():
    """
    Figure 2: ResNet-Rollout Architecture Diagram
    
    Shows:
    - Input: 4 latent frames stacked
    - Model: ResNet-Rollout (~1M params)
    - Training: Teacher-less rollout loop
    - Output: Sharp latent frame
    """
    
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # ========================================================================
    # INPUT: 4 Latent Frames (Stacked)
    # ========================================================================
    
    input_x = 1.5
    input_y = 5
    frame_spacing = 0.15
    
    # Draw 4 stacked frames
    for i in range(4):
        alpha = 0.4 + 0.2 * (i / 3)
        color_intensity = 0.3 + 0.7 * (i / 3)
        
        frame = FancyBboxPatch((input_x + i*frame_spacing, input_y + i*frame_spacing), 
                               1.2, 1.2,
                               boxstyle="round,pad=0.05", 
                               edgecolor='darkblue', 
                               facecolor=plt.cm.Blues(color_intensity),
                               linewidth=2.5, alpha=alpha, zorder=10-i)
        ax.add_patch(frame)
    
    # Labels
    ax.text(input_x + 0.6, input_y - 0.5, r'$z_{t-3:t}$', fontsize=16, ha='center',
           weight='bold', color='darkblue')
    ax.text(input_x + 0.6, input_y - 1.0, '4 Frames', fontsize=12, ha='center',
           style='italic', color='gray')
    ax.text(input_x + 0.6, input_y - 1.4, r'$\mathbb{R}^{64 \times 1 \times 64 \times 64}$', 
           fontsize=10, ha='center', family='monospace', color='darkblue')
    
    # ========================================================================
    # ARROW: Input → Model
    # ========================================================================
    
    arrow1 = FancyArrowPatch((input_x + 1.8, input_y + 0.9), (4.0, input_y + 0.9),
                            arrowstyle='->', mutation_scale=30, linewidth=3,
                            color='darkblue')
    ax.add_patch(arrow1)
    
    # ========================================================================
    # MODEL: ResNet-Rollout
    # ========================================================================
    
    model_x = 4.5
    model_y = 4
    
    # Main model box
    model_box = FancyBboxPatch((model_x, model_y), 3.5, 4,
                              boxstyle="round,pad=0.1", 
                              edgecolor='darkred', 
                              facecolor='mistyrose',
                              linewidth=4, zorder=5)
    ax.add_patch(model_box)
    
    # Model title
    ax.text(model_x + 1.75, model_y + 3.5, 'ResNet-Rollout', fontsize=16, ha='center',
           weight='bold', color='darkred')
    
    # Architecture details
    components = [
        (model_y + 3.0, 'Conv3d(64→64)', 'steelblue'),
        (model_y + 2.5, 'GroupNorm + SiLU', 'steelblue'),
        (model_y + 1.9, '4× ResBlock(64)', 'red'),
        (model_y + 1.3, 'Conv3d(64→16)', 'steelblue'),
    ]
    
    for y_pos, text, color in components:
        box = Rectangle((model_x + 0.3, y_pos - 0.15), 2.9, 0.35,
                       edgecolor=color, facecolor='white',
                       linewidth=2, zorder=6)
        ax.add_patch(box)
        ax.text(model_x + 1.75, y_pos, text, fontsize=11, ha='center',
               weight='bold', color=color, zorder=7)
    
    # Parameter count
    ax.text(model_x + 1.75, model_y + 0.5, r'$\sim$1M Parameters', fontsize=12, ha='center',
           style='italic', color='darkred',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                    edgecolor='darkred', linewidth=2))
    
    # ========================================================================
    # ARROW: Model → Output
    # ========================================================================
    
    arrow2 = FancyArrowPatch((model_x + 3.5, model_y + 2), (9.5, model_y + 2),
                            arrowstyle='->', mutation_scale=30, linewidth=3,
                            color='darkgreen')
    ax.add_patch(arrow2)
    
    # ========================================================================
    # OUTPUT: Predicted Latent Frame
    # ========================================================================
    
    output_x = 10
    output_y = 5
    
    output_frame = FancyBboxPatch((output_x, output_y), 1.5, 1.5,
                                 boxstyle="round,pad=0.05", 
                                 edgecolor='darkgreen', 
                                 facecolor='lightgreen',
                                 linewidth=3, zorder=10)
    ax.add_patch(output_frame)
    
    # Star for "sharp"
    ax.scatter([output_x + 0.75], [output_y + 0.75], s=800, c='gold', marker='*',
              edgecolors='orange', linewidths=2, zorder=15)
    
    # Labels
    ax.text(output_x + 0.75, output_y - 0.5, r'$z_{t+1}$', fontsize=16, ha='center',
           weight='bold', color='darkgreen')
    ax.text(output_x + 0.75, output_y - 1.0, 'Sharp!', fontsize=12, ha='center',
           style='italic', color='darkgreen', weight='bold')
    ax.text(output_x + 0.75, output_y - 1.4, r'$\mathbb{R}^{16 \times 1 \times 64 \times 64}$', 
           fontsize=10, ha='center', family='monospace', color='darkgreen')
    
    # ========================================================================
    # TEACHER-LESS ROLLOUT LOOP
    # ========================================================================
    
    # Loop arrow (output back to input)
    loop_arrow = FancyArrowPatch((output_x + 0.75, output_y - 1.8), (input_x + 0.6, input_y - 1.8),
                                arrowstyle='->', mutation_scale=30, linewidth=3,
                                color='purple', linestyle='dashed')
    ax.add_patch(loop_arrow)
    
    # Loop label
    ax.text(6, input_y - 2.3, 'Teacher-Less Rollout', fontsize=14, ha='center',
           weight='bold', color='purple',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lavender', 
                    edgecolor='purple', linewidth=2))
    ax.text(6, input_y - 2.9, r'(Prediction $\rightarrow$ Next Input)', fontsize=11, ha='center',
           style='italic', color='purple')
    
    # ========================================================================
    # LOSS FUNCTION
    # ========================================================================
    
    loss_x = 12.5
    loss_y = 5
    
    loss_box = FancyBboxPatch((loss_x, loss_y), 3, 2.5,
                             boxstyle="round,pad=0.1", 
                             edgecolor='darkorange', 
                             facecolor='wheat',
                             linewidth=3, zorder=5)
    ax.add_patch(loss_box)
    
    # Loss title
    ax.text(loss_x + 1.5, loss_y + 2.1, 'Loss Function', fontsize=14, ha='center',
           weight='bold', color='darkorange')
    
    # Loss formula
    ax.text(loss_x + 1.5, loss_y + 1.5, r'$\mathcal{L} = \mathcal{L}_{\text{motion}}$', 
           fontsize=13, ha='center', weight='bold', color='darkred')
    
    ax.text(loss_x + 1.5, loss_y + 0.9, r'Motion-Weighted L1:', fontsize=11, ha='center',
           style='italic', color='gray')
    ax.text(loss_x + 1.5, loss_y + 0.4, r'$w = 10.0$ if moving', fontsize=10, ha='center',
           family='monospace', color='darkred')
    ax.text(loss_x + 1.5, loss_y - 0.05, r'$w = 1.0$ if static', fontsize=10, ha='center',
           family='monospace', color='gray')
    
    # Arrow from output to loss
    arrow3 = FancyArrowPatch((output_x + 1.5, output_y + 0.75), (loss_x, loss_y + 1.25),
                            arrowstyle='->', mutation_scale=25, linewidth=2.5,
                            color='darkorange', linestyle='dotted')
    ax.add_patch(arrow3)
    
    # ========================================================================
    # TRAINING CONFIG
    # ========================================================================
    
    config_x = 12.5
    config_y = 1.5
    
    config_text = [
        'Training Configuration:',
        '• Epochs: 500',
        '• Batch Size: 32',
        '• Optimizer: AdamW (1e-4)',
        '• Context: 4 frames',
        '• Rollout: 4 steps',
    ]
    
    config_box = FancyBboxPatch((config_x, config_y - 0.3), 3, 2.3,
                               boxstyle="round,pad=0.1", 
                               edgecolor='gray', 
                               facecolor='lightgray',
                               linewidth=2, alpha=0.8, zorder=5)
    ax.add_patch(config_box)
    
    y_offset = config_y + 1.6
    for i, text in enumerate(config_text):
        fontsize = 12 if i == 0 else 10
        weight = 'bold' if i == 0 else 'normal'
        ax.text(config_x + 1.5, y_offset - i*0.3, text, fontsize=fontsize, ha='center',
               weight=weight, color='black' if i == 0 else 'darkslategray', zorder=6)
    
    # ========================================================================
    # KEY INSIGHT BOX
    # ========================================================================
    
    insight_x = 1
    insight_y = 0.5
    
    insight_box = FancyBboxPatch((insight_x, insight_y), 10, 1.2,
                                boxstyle="round,pad=0.1", 
                                edgecolor='darkgreen', 
                                facecolor='lightgreen',
                                linewidth=3, alpha=0.9, zorder=5)
    ax.add_patch(insight_box)
    
    ax.text(insight_x + 5, insight_y + 0.8, '✅ Key Insight: Context Window Provides Velocity', 
           fontsize=14, ha='center', weight='bold', color='darkgreen', zorder=6)
    ax.text(insight_x + 5, insight_y + 0.3, 
           r'$P(z_{t+1} | z_{t-3:t}) \gg P(z_{t+1} | z_t)$ → Sharp Predictions (71.08 px RMSE)', 
           fontsize=12, ha='center', style='italic', color='darkgreen', zorder=6)
    
    # ========================================================================
    # TITLE
    # ========================================================================
    
    ax.text(8, 9.5, 'ResNet-Rollout: Simple, Fast, Sharp', fontsize=22, ha='center',
           weight='bold', color='darkred')
    ax.text(8, 9.0, 'Architecture & Training Pipeline', fontsize=16, ha='center',
           style='italic', color='gray')
    
    plt.tight_layout()
    
    # Save
    plt.savefig('figure_2_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('figure_2_architecture.pdf', bbox_inches='tight', facecolor='white')
    print("✓ Saved: figure_2_architecture.png + figure_2_architecture.pdf")
    
    plt.close()


# ============================================================================
# FIGURE 3: PERFORMANCE COMPARISON (BONUS)
# ============================================================================

def generate_figure_3_results():
    """
    Figure 3: Performance Comparison Across All Models
    
    Bar chart showing RMSE and detection rates
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    models = ['Baseline\nResNet', 'U-Net\n(Single)', 'Rollout\n(U-Net)', 
              'Prob\nPlus', 'Sharp\nShooter', 'ResNet\nRollout']
    rmse = [None, 165, 165.20, 77.81, 107.97, 71.08]
    detection = [0, 100, 100, 100, 59, 99]
    colors = ['lightgray', 'lightcoral', 'lightsalmon', 'gold', 'lightblue', 'limegreen']
    
    # RMSE Bar Chart
    valid_models = []
    valid_rmse = []
    valid_colors = []
    for i, (m, r, c) in enumerate(zip(models, rmse, colors)):
        if r is not None:
            valid_models.append(m)
            valid_rmse.append(r)
            valid_colors.append(c)
    
    bars1 = ax1.bar(valid_models, valid_rmse, color=valid_colors, edgecolor='black', linewidth=2)
    ax1.set_ylabel('RMSE (pixels)', fontsize=14, weight='bold')
    ax1.set_title('Tracking Accuracy (Lower is Better)', fontsize=16, weight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 180)
    
    # Add value labels
    for bar, val in zip(bars1, valid_rmse):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=11, weight='bold')
    
    # Highlight winner
    ax1.text(len(valid_models)-1, valid_rmse[-1] - 20, '🥇 WINNER', ha='center',
            fontsize=13, weight='bold', color='darkgreen',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', 
                     edgecolor='darkgreen', linewidth=2))
    
    # Detection Rate Bar Chart
    bars2 = ax2.bar(models, detection, color=colors, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Detection Rate (%)', fontsize=14, weight='bold')
    ax2.set_title('Object Detection Reliability', fontsize=16, weight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 110)
    ax2.axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect (100%)')
    
    # Add value labels
    for bar, val in zip(bars2, detection):
        height = bar.get_height()
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{val}%', ha='center', va='bottom', fontsize=11, weight='bold')
    
    ax2.legend(fontsize=11)
    
    fig.suptitle('Performance Comparison: All Models', fontsize=20, weight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    plt.savefig('figure_3_results.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('figure_3_results.pdf', bbox_inches='tight', facecolor='white')
    print("✓ Saved: figure_3_results.png + figure_3_results.pdf")
    
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("="*80)
    print("\nStyle: CVPR/NeurIPS Academic Standard")
    print("Resolution: 300 DPI")
    print("Formats: PNG + PDF (for LaTeX)")
    print("\n" + "="*80)
    
    print("\n📊 Generating Figure 1: The 'Lucid Dream' Concept...")
    generate_figure_1_concept()
    
    print("\n🏗️  Generating Figure 2: ResNet-Rollout Architecture...")
    generate_figure_2_architecture()
    
    print("\n📈 Generating Figure 3: Performance Comparison...")
    generate_figure_3_results()
    
    print("\n" + "="*80)
    print("🎉 ALL FIGURES GENERATED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated Files:")
    print("  • figure_1_concept.png + .pdf")
    print("  • figure_2_architecture.png + .pdf")
    print("  • figure_3_results.png + .pdf")
    print("\n📝 LaTeX Usage:")
    print("  \\includegraphics[width=\\textwidth]{figure_1_concept.pdf}")
    print("  \\includegraphics[width=\\textwidth]{figure_2_architecture.pdf}")
    print("  \\includegraphics[width=0.8\\textwidth]{figure_3_results.pdf}")
    print("="*80)


if __name__ == "__main__":
    main()

