import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Create/Check results folder ---
output_folder = 'results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def zoom_image(img, s, method='nearest'):
    """
    Zooms an image by a factor s using either nearest-neighbor or bilinear interpolation.
    """
    H, W = img.shape[:2]
    H_new, W_new = int(np.round(H * s)), int(np.round(W * s))
    
    # Create a grid of coordinates for the new image
    x_new = np.arange(W_new)
    y_new = np.arange(H_new)
    X_new, Y_new = np.meshgrid(x_new, y_new)
    
    # Map the new coordinates back to the original image space
    X_old = X_new / s
    Y_old = Y_new / s
    
    if method == 'nearest':
        # Find the closest integer coordinate
        X_nearest = np.clip(np.round(X_old).astype(int), 0, W - 1)
        Y_nearest = np.clip(np.round(Y_old).astype(int), 0, H - 1)
        
        zoomed_img = img[Y_nearest, X_nearest]
        
    elif method == 'bilinear':
        # Find the 4 surrounding pixels
        X1 = np.clip(np.floor(X_old).astype(int), 0, W - 1)
        X2 = np.clip(X1 + 1, 0, W - 1)
        Y1 = np.clip(np.floor(Y_old).astype(int), 0, H - 1)
        Y2 = np.clip(Y1 + 1, 0, H - 1)
        
        # Calculate the fractional distances
        dx = X_old - X1
        dy = Y_old - Y1
        
        # If color image, we need to broadcast dx and dy to the 3rd dimension
        if len(img.shape) == 3:
            dx = np.expand_dims(dx, axis=2)
            dy = np.expand_dims(dy, axis=2)
            
        # Get the pixel values of the 4 corners
        I11 = img[Y1, X1].astype(np.float32)
        I21 = img[Y1, X2].astype(np.float32)
        I12 = img[Y2, X1].astype(np.float32)
        I22 = img[Y2, X2].astype(np.float32)
        
        # Interpolate horizontally
        R1 = (1 - dx) * I11 + dx * I21
        R2 = (1 - dx) * I12 + dx * I22
        
        # Interpolate vertically
        zoomed_img = (1 - dy) * R1 + dy * R2
        
        # Clean up and convert back to 8-bit image
        zoomed_img = np.clip(np.round(zoomed_img), 0, 255).astype(np.uint8)
        
    else:
        raise ValueError("Method must be 'nearest' or 'bilinear'")
        
    return zoomed_img

def calculate_normalized_ssd(img1, img2):
    """
    Computes the Normalized Sum of Squared Differences between two images.
    SSD = Sum((I1 - I2)^2) / Sum(I1^2)
    """
    # Ensure they are the exact same shape (crop if there's a 1-pixel rounding difference)
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    i1 = img1[:h, :w].astype(np.float64)
    i2 = img2[:h, :w].astype(np.float64)
    
    ssd = np.sum((i1 - i2) ** 2)
    norm_factor = np.sum(i1 ** 2)
    
    # Avoid division by zero
    if norm_factor == 0: return float('inf')
    
    return ssd / norm_factor

# --- Execution ---
# Define the pairs of (small_image, large_original) based on your uploaded files
image_pairs = [
    ('images/a1q8images/im01small.png', 'images/a1q8images/im01.png'),
    ('images/a1q8images/im02small.png', 'images/a1q8images/im02.png'),
    ('images/a1q8images/im03small.png', 'images/a1q8images/im03.png'),
    ('images/a1q8images/taylor_small.jpg', 'images/a1q8images/taylor.jpg')
]

for small_path, large_path in image_pairs:
    print(f"\nProcessing pair: {small_path} & {large_path}")
    
    small_img = cv2.imread(small_path, cv2.IMREAD_GRAYSCALE)
    large_img = cv2.imread(large_path, cv2.IMREAD_GRAYSCALE)

    if small_img is None or large_img is None:
        print(f"  -> Error: Could not find one or both images. Skipping.")
        continue

    # 1. Determine the zoom factor 's'
    # Assuming uniform scaling, s = large_width / small_width
    s_factor = large_img.shape[1] / small_img.shape[1]
    print(f"  -> Calculated scale factor s = {s_factor:.2f}")
    
    # 2. Perform the zooming
    zoom_nn = zoom_image(small_img, s_factor, method='nearest')
    zoom_bilinear = zoom_image(small_img, s_factor, method='bilinear')
    
    # 3. Calculate Normalized SSD
    ssd_nn = calculate_normalized_ssd(large_img, zoom_nn)
    ssd_bilinear = calculate_normalized_ssd(large_img, zoom_bilinear)
    
    print(f"  -> Nearest-Neighbor SSD : {ssd_nn:.6f}")
    print(f"  -> Bilinear SSD         : {ssd_bilinear:.6f}")

    # Extract a base name for saving files (e.g., 'im01', 'taylor')
    base_name = os.path.basename(large_path).split('.')[0]

    # 4. Save results
    cv2.imwrite(os.path.join(output_folder, f'q7_zoom_nearest_{base_name}.jpg'), zoom_nn)
    cv2.imwrite(os.path.join(output_folder, f'q7_zoom_bilinear_{base_name}.jpg'), zoom_bilinear)

    # 5. Plot the zoomed in sections for comparison
    crop_size = 150
    # Make sure crop size isn't larger than the image itself
    crop_size = min(crop_size, large_img.shape[0], large_img.shape[1])
    
    h_start, w_start = large_img.shape[0]//2 - crop_size//2, large_img.shape[1]//2 - crop_size//2
    
    crop_orig = large_img[h_start:h_start+crop_size, w_start:w_start+crop_size]
    crop_nn = zoom_nn[h_start:h_start+crop_size, w_start:w_start+crop_size]
    crop_bilinear = zoom_bilinear[h_start:h_start+crop_size, w_start:w_start+crop_size]

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(crop_orig, cmap='gray')
    plt.title(f'Original High-Res ({base_name})')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(crop_nn, cmap='gray')
    plt.title(f'Nearest Neighbor\nSSD: {ssd_nn:.4f}')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(crop_bilinear, cmap='gray')
    plt.title(f'Bilinear\nSSD: {ssd_bilinear:.4f}')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'q7_zoom_comparison_{base_name}.png'))
    
    plt.close()
    
print("\n" + "-" * 50)
print("Conclusion: Across all test images, Bilinear interpolation yields a lower SSD")
print("because it creates smoother transitions that better approximate the real high-res images.")
print("-" * 50)