"""
Segmentation Metrics for Validation

Compare predicted structures to ground truth masks.
Implements standard metrics: Dice, IoU, Hausdorff distance, etc.
"""

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import label, find_objects
from skimage.metrics import structural_similarity as ssim
from typing import Dict, Tuple, Optional
import cv2


class SegmentationMetrics:
    """Calculate segmentation quality metrics."""
    
    @staticmethod
    def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate Dice coefficient (F1 score for segmentation).
        
        Dice = 2|A ∩ B| / (|A| + |B|)
        
        Args:
            pred: Predicted binary mask
            gt: Ground truth binary mask
            
        Returns:
            Dice coefficient in [0, 1]
        """
        pred = pred.astype(bool)
        gt = gt.astype(bool)
        
        intersection = np.sum(pred & gt)
        union = np.sum(pred) + np.sum(gt)
        
        if union == 0:
            return 1.0  # Both empty
        
        return 2.0 * intersection / union
    
    @staticmethod
    def iou(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU / Jaccard index).
        
        IoU = |A ∩ B| / |A ∪ B|
        
        Args:
            pred: Predicted binary mask
            gt: Ground truth binary mask
            
        Returns:
            IoU in [0, 1]
        """
        pred = pred.astype(bool)
        gt = gt.astype(bool)
        
        intersection = np.sum(pred & gt)
        union = np.sum(pred | gt)
        
        if union == 0:
            return 1.0  # Both empty
        
        return intersection / union
    
    @staticmethod
    def hausdorff_distance(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate Hausdorff distance between boundaries.
        
        Measures maximum distance from any point on one boundary
        to the nearest point on the other boundary.
        
        Args:
            pred: Predicted binary mask
            gt: Ground truth binary mask
            
        Returns:
            Hausdorff distance in pixels
        """
        # Extract boundaries
        pred_boundary = SegmentationMetrics._get_boundary(pred)
        gt_boundary = SegmentationMetrics._get_boundary(gt)
        
        if len(pred_boundary) == 0 or len(gt_boundary) == 0:
            return np.inf
        
        # Directed Hausdorff distances
        d1 = directed_hausdorff(pred_boundary, gt_boundary)[0]
        d2 = directed_hausdorff(gt_boundary, pred_boundary)[0]
        
        # Symmetric Hausdorff
        return max(d1, d2)
    
    @staticmethod
    def _get_boundary(mask: np.ndarray) -> np.ndarray:
        """Extract boundary coordinates from binary mask."""
        # Use morphological operations
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        boundary = mask.astype(np.uint8) - eroded
        
        # Get coordinates
        coords = np.column_stack(np.where(boundary > 0))
        return coords
    
    @staticmethod
    def pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate pixel-wise accuracy.
        
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        
        Args:
            pred: Predicted binary mask
            gt: Ground truth binary mask
            
        Returns:
            Pixel accuracy in [0, 1]
        """
        pred = pred.astype(bool)
        gt = gt.astype(bool)
        
        correct = np.sum(pred == gt)
        total = pred.size
        
        return correct / total
    
    @staticmethod
    def per_object_metrics(
        pred: np.ndarray, 
        gt: np.ndarray
    ) -> Dict:
        """
        Calculate metrics per individual object.
        
        Args:
            pred: Predicted labeled mask (each object has unique ID)
            gt: Ground truth labeled mask
            
        Returns:
            Dictionary with per-object statistics
        """
        pred_labeled, pred_num = label(pred)
        gt_labeled, gt_num = label(gt)
        
        # Match objects (simple: by centroid overlap)
        pred_objects = []
        gt_objects = []
        
        for obj_id in range(1, pred_num + 1):
            obj_mask = pred_labeled == obj_id
            if np.sum(obj_mask) > 0:
                centroid = np.array(np.where(obj_mask)).mean(axis=1)
                pred_objects.append({
                    "id": obj_id,
                    "centroid": centroid,
                    "area": np.sum(obj_mask),
                })
        
        for obj_id in range(1, gt_num + 1):
            obj_mask = gt_labeled == obj_id
            if np.sum(obj_mask) > 0:
                centroid = np.array(np.where(obj_mask)).mean(axis=1)
                gt_objects.append({
                    "id": obj_id,
                    "centroid": centroid,
                    "area": np.sum(obj_mask),
                })
        
        # Match objects (nearest centroid)
        matches = []
        for pred_obj in pred_objects:
            distances = [
                np.linalg.norm(pred_obj["centroid"] - gt_obj["centroid"])
                for gt_obj in gt_objects
            ]
            if distances:
                min_idx = np.argmin(distances)
                matches.append({
                    "pred_id": pred_obj["id"],
                    "gt_id": gt_objects[min_idx]["id"],
                    "distance": distances[min_idx],
                })
        
        return {
            "num_predicted": pred_num,
            "num_ground_truth": gt_num,
            "num_matched": len(matches),
            "matches": matches,
        }
    
    @staticmethod
    def compute_all_metrics(
        pred: np.ndarray, 
        gt: np.ndarray,
        pixel_size_nm: Optional[float] = None
    ) -> Dict:
        """
        Compute all segmentation metrics.
        
        Args:
            pred: Predicted binary mask
            gt: Ground truth binary mask
            pixel_size_nm: Physical pixel size (for distance metrics)
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            "dice": SegmentationMetrics.dice_coefficient(pred, gt),
            "iou": SegmentationMetrics.iou(pred, gt),
            "pixel_accuracy": SegmentationMetrics.pixel_accuracy(pred, gt),
            "hausdorff_distance_pixels": SegmentationMetrics.hausdorff_distance(pred, gt),
        }
        
        # Convert to physical units if pixel size provided
        if pixel_size_nm:
            metrics["hausdorff_distance_nm"] = (
                metrics["hausdorff_distance_pixels"] * pixel_size_nm
            )
        
        # Per-object metrics
        metrics["per_object"] = SegmentationMetrics.per_object_metrics(pred, gt)
        
        return metrics


if __name__ == "__main__":
    # Test with synthetic masks
    size = (256, 256)
    
    # Create ground truth: circle
    gt = np.zeros(size, dtype=bool)
    center = (128, 128)
    radius = 50
    y, x = np.ogrid[:size[0], :size[1]]
    gt[(x - center[0])**2 + (y - center[1])**2 <= radius**2] = True
    
    # Create prediction: slightly shifted circle
    pred = np.zeros(size, dtype=bool)
    center_pred = (130, 130)  # Slight shift
    pred[(x - center_pred[0])**2 + (y - center_pred[1])**2 <= radius**2] = True
    
    # Compute metrics
    metrics = SegmentationMetrics.compute_all_metrics(pred, gt, pixel_size_nm=100.0)
    
    print("Segmentation Metrics:")
    print(f"  Dice coefficient: {metrics['dice']:.4f}")
    print(f"  IoU: {metrics['iou']:.4f}")
    print(f"  Pixel accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"  Hausdorff distance: {metrics['hausdorff_distance_pixels']:.2f} pixels")
    print(f"  Hausdorff distance: {metrics['hausdorff_distance_nm']:.2f} nm")
