import cv2
import numpy as np

method_list = ['Otsu','Adaptive threshold','Watershed','K-means','Region Growing']

def segment_organ_classical(image, method='otsu', **kwargs):
    """
    Segmentare folosind tehnici clasice de Computer Vision
    
    Metode disponibile:
    - 'adaptive_threshold': Praguri adaptive (bun pentru contrast variabil)
    - 'otsu': Metoda Otsu (bun pentru contrast bine definit)
    - 'watershed': Segmentare bazată pe watershed
    """
    print(f"Se aplică segmentarea: {method}...")
    
    if method == 'Otsu':
        _, mask = cv2.threshold(
            image, 0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif method == 'Adaptive threshold':
        mask = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            51, 15
        )    
    elif method == 'Watershed':
        # 1. Otsu threshold 
        ret, bin_img = cv2.threshold(
            image, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 2. Noise removal (morph open)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=2)

        # 3. Sure background
        sure_bg = cv2.dilate(clean, kernel, iterations=3)

        # 4. Distance transform for foreground
        dist = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
        sure_fg = sure_fg.astype(np.uint8)

        # 5. Unknown region = background - foreground
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 6. Create markers
        ret, markers = cv2.connectedComponents(sure_fg)

        # Increase markers by +1 so background becomes 1 instead of 0
        markers = markers + 1

        # Set unknown to 0 (watershed will fill them)
        markers[unknown == 255] = 0

        # 7. Watershed needs a 3-channel image
        img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color, markers)

        # 8. Final mask: watershed boundaries = -1 → remove them
        mask = np.uint8(markers > 1) * 255
    elif method == 'K-means':
        # Convert grayscale → RGB (K-means expects 3-channel data)
        imageRGB = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Reshape to a list of pixels: (H*W, 3)
        pixel_values = imageRGB.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        # K-means criteria: stop at 100 iterations OR ε < 0.2
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                    100, 0.2)

        # Number of clusters (foreground/background)
        K = 3

        # Apply K-means
        _, labels, centers = cv2.kmeans(
            pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        # Labels returned as (num_pixels, 1) → reshape to image shape
        labels = labels.flatten().reshape(image.shape)

        # Determine which cluster is foreground:
        # usually the darker cluster (lower intensity)
        centers_gray = np.mean(centers, axis=1)
        fg_cluster = np.argmin(centers_gray)   # darkest cluster

        # Build binary mask
        mask = np.uint8(labels == fg_cluster) * 255
        mask = cv2.bitwise_not(mask)

    elif method == 'Region Growing':
         # Get parameters or use defaults
        threshold = kwargs.get('threshold', 20)  # Diferență maximă de intensitate
        seeds = kwargs.get('seeds', None)  # Seed points (list of (x, y) tuples)
        
        # Initialize mask
        h, w = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        visited = np.zeros((h, w), dtype=bool)
        
        # If no seeds provided, auto-select seeds from center region
        if seeds is None:
            # Select multiple seeds from BRIGHTER regions in center (the organ)
            center_region = image[h//4:3*h//4, w//4:3*w//4]
            mean_val = np.mean(center_region)
            
            # Find pixels BRIGHTER than mean in center region (organ is typically brighter)
            y_coords, x_coords = np.where(center_region > mean_val)
            
            # Select up to 10 seed points
            num_seeds = min(10, len(y_coords))
            if num_seeds > 0:
                indices = np.random.choice(len(y_coords), num_seeds, replace=False)
                seeds = [(x_coords[i] + w//4, y_coords[i] + h//4) for i in indices]
            else:
                # Fallback to center point
                seeds = [(w//2, h//2)]
        
        # Region growing from each seed
        for seed in seeds:
            seed_x, seed_y = seed
            
            # Skip if already visited or out of bounds
            if (seed_y < 0 or seed_y >= h or seed_x < 0 or seed_x >= w or 
                visited[seed_y, seed_x]):
                continue
            
            # Queue for BFS
            queue = [(seed_x, seed_y)]
            seed_value = float(image[seed_y, seed_x])
            
            # 8-connectivity neighbors
            neighbors = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),           (0, 1),
                        (1, -1),  (1, 0),  (1, 1)]
            
            while queue:
                x, y = queue.pop(0)
                
                if visited[y, x]:
                    continue
                
                # Check if pixel is similar to seed
                if abs(float(image[y, x]) - seed_value) <= threshold:
                    mask[y, x] = 255
                    visited[y, x] = True
                    
                    # Add neighbors to queue
                    for dx, dy in neighbors:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < w and 0 <= ny < h and not visited[ny, nx]):
                            queue.append((nx, ny))
        
        # Optional: morphological closing to fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    else:
        raise ValueError(f"Metoda necunoscuta: {method}")
    
    return mask