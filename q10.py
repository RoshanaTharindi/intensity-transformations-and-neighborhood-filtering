import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Create/Check results folder ---
output_folder = 'results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ==========================================
# (a) Manual Bilateral Filter Function
# ==========================================
def manual_bilateral_filter(image, d, sigma_s, sigma_r):
    """
    Manually implements a bilateral filter for grayscale images.
    """
    print("Starting manual bilateral filtering...")
    img_float = image.astype(np.float32)
    h, w = img_float.shape
    result = np.zeros_like(img_float)
    
    radius = d // 2
    
    # Precompute the spatial Gaussian weights (these are constant for every pixel's neighborhood)
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    spatial_weights = np.exp(-(xx**2 + yy**2) / (2 * (sigma_s**2)))
    
    # Pad the image to handle edge cases (pixels near the border of the image)
    padded_img = np.pad(img_float, radius, mode='reflect')
    
    # Iterate over every single pixel in the image
    for i in range(h):
        for j in range(w):
            # Extract the local neighborhood window
            region = padded_img[i:i+d, j:j+d]
            
            # The intensity of the center pixel
            center_intensity = img_float[i, j]
            
            # Calculate the range weights based on intensity differences
            intensity_diffs = region - center_intensity
            range_weights = np.exp(-(intensity_diffs**2) / (2 * (sigma_r**2)))
            
            # Combine spatial and range weights
            bilateral_weights = spatial_weights * range_weights
            
            # Normalize the weights and compute the new pixel value
            weight_sum = np.sum(bilateral_weights)
            filtered_pixel = np.sum(region * bilateral_weights) / weight_sum
            
            result[i, j] = filtered_pixel
            
    print("Manual bilateral filtering complete!")
    return np.clip(result, 0, 255).astype(np.uint8)

# --- Load Image ---
img_path = 'images/a1q8images/taylor.jpg'
img_bgr = cv2.imread(img_path)

if img_bgr is None:
    print(f"Error: Could not load '{img_path}'. Please check the filename!")
else:
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # --- Performance Optimization ---
    scale_percent = 400 / img_gray.shape[1] # Scale width down to 400 pixels
    new_width = int(img_gray.shape[1] * scale_percent)
    new_height = int(img_gray.shape[0] * scale_percent)
    img_gray = cv2.resize(img_gray, (new_width, new_height))

    # --- Filter Parameters ---
    d = 7           # Kernel diameter
    sigma_s = 15.0  # Spatial standard deviation
    sigma_r = 40.0  # Range (intensity) standard deviation

    # ==========================================
    # (b) Apply Gaussian Smoothing
    # ==========================================
    gaussian_smoothed = cv2.GaussianBlur(img_gray, (d, d), sigma_s)

    # ==========================================
    # (c) Apply OpenCV's Bilateral Filter
    # ==========================================
    # cv2.bilateralFilter takes (image, d, sigmaColor, sigmaSpace)
    opencv_bilateral = cv2.bilateralFilter(img_gray, d, sigma_r, sigma_s)

    # ==========================================
    # (d) Apply Manual Bilateral Filter
    # ==========================================
    manual_bilateral = manual_bilateral_filter(img_gray, d, sigma_s, sigma_r)

    # --- Save the results ---
    cv2.imwrite(os.path.join(output_folder, 'q10_gaussian.jpg'), gaussian_smoothed)
    cv2.imwrite(os.path.join(output_folder, 'q10_opencv_bilateral.jpg'), opencv_bilateral)
    cv2.imwrite(os.path.join(output_folder, 'q10_manual_bilateral.jpg'), manual_bilateral)
    print(f"Processed images saved to '{output_folder}' folder.")

    # --- Plot the Results ---
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Original Grayscale Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(gaussian_smoothed, cmap='gray')
    plt.title(f'Gaussian Smoothing ($d={d}, \sigma={sigma_s}$)\n(Blurs edges)')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(opencv_bilateral, cmap='gray')
    plt.title(f"OpenCV Bilateral ($\sigma_s={sigma_s}, \sigma_r={sigma_r}$)\n(Preserves edges)")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(manual_bilateral, cmap='gray')
    plt.title(f"Manual Bilateral ($\sigma_s={sigma_s}, \sigma_r={sigma_r}$)\n(Matches OpenCV)")
    plt.axis('off')

    plt.tight_layout()
    plot_save_path = os.path.join(output_folder, 'q10_bilateral_comparison.png')
    plt.savefig(plot_save_path)

    # Show the plot window
    plt.show()