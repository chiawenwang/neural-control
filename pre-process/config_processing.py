### Pre-processing pictures
### Using tools on https://automeris.io/ to extract dots along the elastic beams from real-world pictures
### This programs constructs the elastic beam shape from dots and converts it to a Nx2 matrix

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import pdist, squareform

def calculate_median_step(points):
    """
    Calculate the 'normal' distance between adjacent points (median step).
    Used to dynamically set thresholds for different image resolutions.
    """
    if len(points) < 2:
        return 5.0
    
    sample = points[:200]
    dists = []
    for p in sample:
        ds = np.linalg.norm(points - p, axis=1)
        ds = ds[ds > 0]
        if len(ds) > 0:
            dists.append(np.min(ds))
            
    if not dists:
        return 5.0
    
    return np.median(dists)

def get_path_with_angle_check(points, start_index, median_step, max_jump_factor=15.0):
    """
    Path trace with directional constraints.
    1. Allow large jumps
    2. BUT, if a jump is large, the direction must remain consistent.
       (Prevents 90-degree turns).
    """
    path = [points[start_index]]
    unvisited = np.delete(points, start_index, axis=0)
    
    max_jump_dist = median_step * max_jump_factor
    # Threshold to trigger angle check (e.g., 3x normal step)
    gap_jump_threshold = median_step * 3.0
    
    while len(unvisited) > 0:
        current_pos = path[-1]
        # Find nearest neighbor
        dists = np.linalg.norm(unvisited - current_pos, axis=1)
        nearest_idx = np.argmin(dists)
        min_dist = dists[nearest_idx]
        candidate_point = unvisited[nearest_idx]
        # 1. Distance hard cutoff
        if min_dist > max_jump_dist:
            break
        # 2. When to stop 
        # If jump is large (>3x step), check if direction changes drastically
        if len(path) > 1 and min_dist > gap_jump_threshold:
            prev_pos = path[-2]
            vec_current = current_pos - prev_pos
            vec_next = candidate_point - current_pos
            norm_curr = np.linalg.norm(vec_current)
            norm_next = np.linalg.norm(vec_next)
            
            if norm_curr > 0 and norm_next > 0:
                # Cosine similarity
                cos_angle = np.dot(vec_current, vec_next) / (norm_curr * norm_next)
                # If angle > 60 deg (cos < 0.5) AND it's a large jump -> Stop
                if cos_angle < 0.5: 
                    break
        # Accept point
        path.append(candidate_point)
        unvisited = np.delete(unvisited, nearest_idx, axis=0)
        
    return np.array(path)

def process_curve_configuration(image_path, real_width_cm, real_height_cm, target_points=50):
    # --- 1. Load and Preprocess ---
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Red color masks
    lower1, upper1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower2, upper2 = np.array([170, 70, 50]), np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    # Close gaps
    kernel = np.ones((5,5), np.uint8) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) 
    # Get centroids
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 5:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                points.append([M["m10"] / M["m00"], M["m01"] / M["m00"]])
    
    pixel_points = np.array(points)
    if len(pixel_points) < 2:
        raise ValueError("Not enough red points detected.")

    # --- 2. Path Tracing ---
    median_step = calculate_median_step(pixel_points)

    D = squareform(pdist(pixel_points))
    i, j = np.unravel_index(D.argmax(), D.shape)
    
    # Run tracer from both ends
    path_from_i = get_path_with_angle_check(pixel_points, i, median_step, max_jump_factor=15.0)
    path_from_j = get_path_with_angle_check(pixel_points, j, median_step, max_jump_factor=15.0)
    
    # Select longest path
    if len(path_from_i) >= len(path_from_j):
        sorted_points = path_from_i
    else:
        sorted_points = path_from_j

    # Standardize direction: Left to Right
    if sorted_points[0][0] > sorted_points[-1][0]:
        sorted_points = sorted_points[::-1]

    # --- 3. Spline Fitting & Resampling ---
    try:
        tck, u = splprep(sorted_points.T, s=10, k=min(3, len(sorted_points)-1))
    except Exception:
        tck, u = splprep(sorted_points.T, s=0, k=1)

    u_fine = np.linspace(0, 1, 2000)
    x_fine, y_fine = splev(u_fine, tck)
    
    diffs = np.hypot(np.diff(x_fine), np.diff(y_fine))
    cumulative_dist = np.concatenate(([0], np.cumsum(diffs)))
    total_len_px = cumulative_dist[-1]
    
    target_dists = np.linspace(0, total_len_px, target_points)
    x_uniform = np.interp(target_dists, cumulative_dist, x_fine)
    y_uniform = np.interp(target_dists, cumulative_dist, y_fine)

    # --- 4. Coordinate Mapping & Arc Length ---
    min_px_x, max_px_x = np.min(x_uniform), np.max(x_uniform)
    min_px_y, max_px_y = np.min(y_uniform), np.max(y_uniform)
    
    scale_x = real_width_cm / (max_px_x - min_px_x) if (max_px_x - min_px_x) > 0 else 1
    scale_y = real_height_cm / (max_px_y - min_px_y) if (max_px_y - min_px_y) > 0 else 1

    final_x = (x_uniform - min_px_x) * scale_x
    final_y = (max_px_y - y_uniform) * scale_y 

    # Calculate Arc Length
    x_fine_cm = (x_fine - min_px_x) * scale_x
    y_fine_cm = (max_px_y - y_fine) * scale_y
    fine_diffs = np.hypot(np.diff(x_fine_cm), np.diff(y_fine_cm))
    arc_length_cm = np.sum(fine_diffs)

    # --- 5. Output ---
    header = (
        f"# {target_points} x 2 (x_cm, y_cm)\n"
        f"# Origin: bottom-left\n"
        f"# width_cm={real_width_cm:.4f} height_cm={real_height_cm:.4f} arc_length_cm={arc_length_cm:.4f}"
    )
    print(header)
    print()

    output_data = np.column_stack((final_x, final_y))
    for row in output_data:
        x_val = 0.0 if abs(row[0]) < 1e-4 else row[0]
        y_val = 0.0 if abs(row[1]) < 1e-4 else row[1]
        print(f"{x_val:.6f}\t{y_val:.6f}")
    
    return output_data

# Example configurations:
# process_curve_configuration("config_1.png", 5.5, 20.5)
# process_curve_configuration("config_2.png", 16.0, 13.0)
# process_curve_configuration("config_3.png", 16.5, 5.5)
# process_curve_configuration("config_4.png", 22.0, 3.0)
# process_curve_configuration("config_5.png", 18.0, 6.5)
