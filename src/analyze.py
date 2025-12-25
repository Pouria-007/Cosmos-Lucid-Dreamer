#!/usr/bin/env python3
"""
Centroid Tracking Analysis for Cosmos-Isaac World Model

Quantitatively validates that the dynamics model correctly predicts object trajectories
by comparing centroids of the red cube in ground truth vs. dreamed frames.

Usage:
    python analyze_tracking.py --ground_truth cosmic_data/frames --dreamed cosmic_results_motion
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Tuple, Optional

# Use high-quality backend for paper figures
matplotlib.use('Agg')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10


def detect_red_centroid(image_path: str, is_dreamed: bool = False, debug: bool = False) -> Optional[Tuple[float, float]]:
    """
    Detect the centroid of the red object in an image using HSV color masking.
    
    Args:
        image_path: Path to the image file
        is_dreamed: If True, use more lenient thresholds for desaturated images
        debug: If True, show debug visualizations
    
    Returns:
        (x, y) centroid coordinates, or None if no red object detected
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not read {image_path}")
        return None
    
    # Resize to 512x512 if needed (for consistent coordinate space)
    if img.shape[0] != 512 or img.shape[1] != 512:
        img = cv2.resize(img, (512, 512))
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Adaptive thresholds based on image type
    if is_dreamed:
        # More lenient thresholds for blurry/desaturated dreamed images
        # Lower saturation and value requirements
        lower_red1 = np.array([0, 0, 100])  # Very low saturation OK
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 0, 100])
        upper_red2 = np.array([180, 255, 255])
    else:
        # Standard thresholds for ground truth
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
    
    # Create masks for red color ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Combine masks
    mask = mask1 | mask2
    
    # Apply morphological operations to clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        # Try alternative: use red channel directly (for very desaturated images)
        if is_dreamed:
            # Split BGR channels
            b, g, r = cv2.split(img)
            # Red object should have r > (g+b)/2
            red_emphasis = r.astype(np.float32) - (g.astype(np.float32) + b.astype(np.float32)) / 2.0
            red_emphasis = np.clip(red_emphasis, 0, 255).astype(np.uint8)
            
            # Threshold
            _, mask = cv2.threshold(red_emphasis, 10, 255, cv2.THRESH_BINARY)
            
            # Morphology
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours again
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                return None
    
    # Find the largest contour (should be the cube)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Filter out very small contours (noise)
    min_area = 100  # pixels
    if cv2.contourArea(largest_contour) < min_area:
        return None
    
    # Calculate moments to find centroid
    M = cv2.moments(largest_contour)
    
    if M["m00"] == 0:
        return None
    
    # Calculate centroid
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    
    if debug:
        # Draw contour and centroid for debugging
        debug_img = img.copy()
        cv2.drawContours(debug_img, [largest_contour], -1, (0, 255, 0), 2)
        cv2.circle(debug_img, (int(cx), int(cy)), 5, (255, 0, 0), -1)
        cv2.imshow("Debug", debug_img)
        cv2.imshow("Mask", mask)
        cv2.waitKey(1)
    
    return (cx, cy)


def load_trajectory(frames_dir: str, frame_pattern: str, is_dreamed: bool = False, max_frames: int = None) -> List[Optional[Tuple[float, float]]]:
    """
    Load trajectory by detecting centroids in all frames.
    
    Args:
        frames_dir: Directory containing frames
        frame_pattern: Pattern for frame filenames (e.g., "frame_%04d.png" or "dream_%02d.png")
        is_dreamed: If True, use lenient thresholds for desaturated images
        max_frames: Maximum number of frames to load
    
    Returns:
        List of (x, y) centroids, or None for frames where detection failed
    """
    frames_path = Path(frames_dir)
    trajectory = []
    
    frame_idx = 0
    while True:
        if max_frames and frame_idx >= max_frames:
            break
        
        # Try to find the frame
        if "%04d" in frame_pattern:
            frame_file = frames_path / (frame_pattern % frame_idx)
        elif "%02d" in frame_pattern:
            frame_file = frames_path / (frame_pattern % frame_idx)
        else:
            frame_file = frames_path / frame_pattern.format(frame_idx)
        
        if not frame_file.exists():
            break
        
        # Detect centroid
        centroid = detect_red_centroid(str(frame_file), is_dreamed=is_dreamed)
        trajectory.append(centroid)
        
        frame_idx += 1
    
    return trajectory


def calculate_trajectory_metrics(
    ground_truth: List[Optional[Tuple[float, float]]],
    dreamed: List[Optional[Tuple[float, float]]]
) -> dict:
    """
    Calculate quantitative metrics comparing ground truth and dreamed trajectories.
    
    Args:
        ground_truth: Ground truth centroids
        dreamed: Dreamed centroids
    
    Returns:
        Dictionary of metrics
    """
    # Filter out None values and align lengths
    min_len = min(len(ground_truth), len(dreamed))
    
    valid_pairs = []
    errors = []
    
    for i in range(min_len):
        gt = ground_truth[i]
        dr = dreamed[i]
        
        if gt is not None and dr is not None:
            valid_pairs.append((gt, dr))
            
            # Calculate Euclidean distance
            error = np.sqrt((gt[0] - dr[0])**2 + (gt[1] - dr[1])**2)
            errors.append(error)
    
    if len(errors) == 0:
        return {
            "mean_error": float('inf'),
            "std_error": 0.0,
            "max_error": float('inf'),
            "min_error": float('inf'),
            "mse": float('inf'),
            "valid_frames": 0,
            "total_frames": min_len,
        }
    
    errors = np.array(errors)
    
    return {
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "max_error": float(np.max(errors)),
        "min_error": float(np.min(errors)),
        "mse": float(np.mean(errors**2)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "valid_frames": len(errors),
        "total_frames": min_len,
        "errors": errors.tolist(),
    }


def plot_tracking_analysis(
    ground_truth: List[Optional[Tuple[float, float]]],
    dreamed: List[Optional[Tuple[float, float]]],
    metrics: dict,
    output_path: str = "tracking_analysis.png"
):
    """
    Create publication-quality figure showing trajectory comparison and error analysis.
    
    Args:
        ground_truth: Ground truth centroids
        dreamed: Dreamed centroids
        metrics: Trajectory metrics
        output_path: Where to save the figure
    """
    # Filter out None values
    gt_clean = [(i, x, y) for i, (x, y) in enumerate(ground_truth) if (x, y) != (None, None)]
    dr_clean = [(i, x, y) for i, (x, y) in enumerate(dreamed) if (x, y) != (None, None)]
    
    if len(gt_clean) == 0 or len(dr_clean) == 0:
        print("Error: No valid centroids detected in trajectories")
        return
    
    # Extract coordinates
    gt_frames, gt_x, gt_y = zip(*gt_clean)
    dr_frames, dr_x, dr_y = zip(*dr_clean)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Top plot: X-Y Trajectory Map
    ax1.plot(gt_x, gt_y, 'o-', color='#2E86AB', linewidth=2, markersize=4, 
             label='Ground Truth', alpha=0.8, markeredgecolor='white', markeredgewidth=0.5)
    ax1.plot(dr_x, dr_y, 's-', color='#A23B72', linewidth=2, markersize=4, 
             label='Dreamed (Predicted)', alpha=0.8, markeredgecolor='white', markeredgewidth=0.5)
    
    # Mark start and end points
    ax1.plot(gt_x[0], gt_y[0], 'o', color='green', markersize=12, 
             label='Start', markeredgecolor='white', markeredgewidth=2)
    ax1.plot(gt_x[-1], gt_y[-1], 'o', color='red', markersize=12, 
             label='End', markeredgecolor='white', markeredgewidth=2)
    
    ax1.set_xlabel('X Position (pixels)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y Position (pixels)', fontsize=12, fontweight='bold')
    ax1.set_title('Object Trajectory: Ground Truth vs. World Model Prediction', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='best', frameon=True, shadow=True, fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_aspect('equal', adjustable='box')
    ax1.invert_yaxis()  # Invert Y axis to match image coordinates
    
    # Bottom plot: Error over Time
    errors = metrics['errors']
    frames = range(len(errors))
    
    ax2.plot(frames, errors, 'o-', color='#F18F01', linewidth=2, markersize=4,
             markeredgecolor='white', markeredgewidth=0.5)
    ax2.axhline(y=metrics['mean_error'], color='red', linestyle='--', linewidth=2,
                label=f'Mean Error: {metrics["mean_error"]:.2f} px')
    ax2.fill_between(frames, 
                     metrics['mean_error'] - metrics['std_error'],
                     metrics['mean_error'] + metrics['std_error'],
                     alpha=0.2, color='red', label=f'±1 Std: {metrics["std_error"]:.2f} px')
    
    ax2.set_xlabel('Frame Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Centroid Distance Error (pixels)', fontsize=12, fontweight='bold')
    ax2.set_title('Trajectory Prediction Error Over Time', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.legend(loc='best', frameon=True, shadow=True, fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(bottom=0)
    
    # Add metrics text box
    metrics_text = (
        f"RMSE: {metrics['rmse']:.2f} px\n"
        f"Max Error: {metrics['max_error']:.2f} px\n"
        f"Min Error: {metrics['min_error']:.2f} px\n"
        f"Valid Frames: {metrics['valid_frames']}/{metrics['total_frames']}"
    )
    ax2.text(0.98, 0.97, metrics_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved tracking analysis figure: {output_path}")
    plt.close()


def save_metrics(metrics: dict, output_path: str = "tracking_metrics.txt"):
    """
    Save trajectory metrics to a text file.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Where to save the metrics
    """
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("COSMOS-ISAAC WORLD MODEL: TRAJECTORY TRACKING ANALYSIS\n")
        f.write("="*60 + "\n\n")
        
        f.write("Centroid Distance Metrics (pixels):\n")
        f.write("-"*60 + "\n")
        f.write(f"  Mean Error:        {metrics['mean_error']:.4f}\n")
        f.write(f"  Std Deviation:     {metrics['std_error']:.4f}\n")
        f.write(f"  Min Error:         {metrics['min_error']:.4f}\n")
        f.write(f"  Max Error:         {metrics['max_error']:.4f}\n")
        f.write(f"  RMSE:              {metrics['rmse']:.4f}\n")
        f.write(f"  MSE:               {metrics['mse']:.4f}\n")
        f.write("\n")
        
        f.write("Frame Statistics:\n")
        f.write("-"*60 + "\n")
        f.write(f"  Valid Frames:      {metrics['valid_frames']}\n")
        f.write(f"  Total Frames:      {metrics['total_frames']}\n")
        f.write(f"  Detection Rate:    {100 * metrics['valid_frames'] / metrics['total_frames']:.2f}%\n")
        f.write("\n")
        
        f.write("="*60 + "\n")
        f.write("Interpretation:\n")
        f.write("="*60 + "\n")
        f.write("Lower RMSE indicates better trajectory prediction accuracy.\n")
        f.write("For a 512x512 image, RMSE < 50 pixels suggests good tracking.\n")
        f.write("="*60 + "\n")
    
    print(f"✓ Saved metrics: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Centroid Tracking Analysis for Cosmos-Isaac World Model"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        default="cosmic_data/frames",
        help="Directory containing ground truth frames"
    )
    parser.add_argument(
        "--ground_truth_pattern",
        type=str,
        default="frame_%04d.png",
        help="Pattern for ground truth frame filenames"
    )
    parser.add_argument(
        "--dreamed",
        type=str,
        default="cosmic_results_motion",
        help="Directory containing dreamed frames"
    )
    parser.add_argument(
        "--dreamed_pattern",
        type=str,
        default="dream_%02d.png",
        help="Pattern for dreamed frame filenames"
    )
    parser.add_argument(
        "--output_figure",
        type=str,
        default="tracking_analysis.png",
        help="Output path for the analysis figure"
    )
    parser.add_argument(
        "--output_metrics",
        type=str,
        default="tracking_metrics.txt",
        help="Output path for the metrics text file"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to analyze"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("CENTROID TRACKING ANALYSIS")
    print("="*60)
    print(f"Ground Truth: {args.ground_truth}")
    print(f"Dreamed:      {args.dreamed}")
    print("="*60)
    
    # Load trajectories
    print("\n1. Loading ground truth trajectory...")
    ground_truth = load_trajectory(
        args.ground_truth,
        args.ground_truth_pattern,
        is_dreamed=False,
        max_frames=args.max_frames
    )
    gt_valid = sum(1 for c in ground_truth if c is not None)
    print(f"   ✓ Loaded {len(ground_truth)} frames ({gt_valid} with valid detections)")
    
    print("\n2. Loading dreamed trajectory...")
    dreamed = load_trajectory(
        args.dreamed,
        args.dreamed_pattern,
        is_dreamed=True,
        max_frames=args.max_frames
    )
    dr_valid = sum(1 for c in dreamed if c is not None)
    print(f"   ✓ Loaded {len(dreamed)} frames ({dr_valid} with valid detections)")
    
    # Calculate metrics
    print("\n3. Calculating trajectory metrics...")
    metrics = calculate_trajectory_metrics(ground_truth, dreamed)
    
    print(f"\n   Results:")
    print(f"   - Mean Error:  {metrics['mean_error']:.2f} pixels")
    print(f"   - RMSE:        {metrics['rmse']:.2f} pixels")
    print(f"   - Max Error:   {metrics['max_error']:.2f} pixels")
    print(f"   - Valid Frames: {metrics['valid_frames']}/{metrics['total_frames']}")
    
    # Generate figure
    print("\n4. Generating publication figure...")
    plot_tracking_analysis(ground_truth, dreamed, metrics, args.output_figure)
    
    # Save metrics
    print("\n5. Saving metrics...")
    save_metrics(metrics, args.output_metrics)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  - Figure:  {args.output_figure}")
    print(f"  - Metrics: {args.output_metrics}")
    print("="*60)


if __name__ == "__main__":
    main()

