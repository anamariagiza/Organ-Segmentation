import cv2
import numpy as np
from scipy import ndimage


def postprocess_mask(mask, original_shape,n_largest=2):
    """
    Post-processes segmentation mask with comprehensive hole filling
    
    Features:
    - Removes small noise components
    - Single organ: keeps largest region only
    - Fills ALL holes inside the mask (small and large)
    - Smooths contours
    - Resizes to original dimensions
    
    Args:
        mask: Binary or label mask (uint8)
        original_shape: Target shape (H, W)
    
    Returns:
        Processed mask resized to original_shape with all holes filled
    """
    # Noise removal with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    # Opening: removes small white noise
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Closing: fills small holes
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=4
    )
    
    # Keep only the largest component (excluding background label 0)
    if num_labels > 1:
        # Get areas of components, skipping background (index 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        # Get indices of top N components (add 1 to correct for the skip)
        top_indices = np.argsort(areas)[::-1][:n_largest] + 1
        # Keep only pixels belonging to top_indices
        cleaned = np.isin(labels, top_indices).astype(np.uint8) * 255
    else:
        cleaned = np.zeros_like(mask)
    
    
    # Binary fill holes using scipy
    filled_scipy = ndimage.binary_fill_holes(cleaned > 0).astype(np.uint8) * 255
    
    # Use the scipy method as it's more reliable
    cleaned = filled_scipy
    
    
    # Smooth contours with dilation + erosion
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)
    cleaned = cv2.erode(cleaned, kernel, iterations=1)
    
    # Gaussian smoothing for natural boundaries
    cleaned = cv2.GaussianBlur(cleaned, (9, 9), 0)
    _, cleaned = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)
    
    # Resize to original shape
    final_mask = cv2.resize(
        cleaned, (original_shape[1], original_shape[0]), 
        interpolation=cv2.INTER_NEAREST
    )
    
    print(f"✓ Post-processing complete: {final_mask.shape}, "
          f"unique values: {np.unique(final_mask)}, "
          f"holes filled: Yes")
    
    return final_mask