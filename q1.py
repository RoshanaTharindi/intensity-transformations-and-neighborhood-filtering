import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Create a folder to save results ---
output_folder = 'results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created directory: {output_folder}")

# --- 1. Load and Normalize the Image ---
image_path = 'images/runway.png' 
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Could not find '{image_path}'.")
else:
    r = img.astype(np.float32) / 255.0

    # --- (a) Gamma correction with gamma = 0.5 ---
    gamma_a = 0.5
    img_gamma_a = np.power(r, gamma_a)

    # --- (b) Gamma correction with gamma = 2.0 ---
    gamma_b = 2.0
    img_gamma_b = np.power(r, gamma_b)

    # --- (c) Contrast Stretching (linear piecewise transformation) ---
    r1 = 0.2
    r2 = 0.8
    s = np.zeros_like(r)
    
    mask_mid = (r >= r1) & (r <= r2)
    s[mask_mid] = (r[mask_mid] - r1) / (r2 - r1)
    
    mask_high = (r > r2)
    s[mask_high] = 1.0

    # Save the individual images to the results folder ---
    # We have to multiply by 255 and convert back to 8-bit integers before saving
    cv2.imwrite(os.path.join(output_folder, 'q1_gamma_0.5.png'), (img_gamma_a * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_folder, 'q1_gamma_2.0.png'), (img_gamma_b * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_folder, 'q1_contrast_stretch.png'), (s * 255).astype(np.uint8))
    print(f"Individual images saved to '{output_folder}' folder!")

    # --- Plot the Results ---
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(r, cmap='gray', vmin=0, vmax=1)
    plt.title('Original Runway Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(img_gamma_a, cmap='gray', vmin=0, vmax=1)
    plt.title('Gamma Correction ($\gamma=0.5$)')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(img_gamma_b, cmap='gray', vmin=0, vmax=1)
    plt.title('Gamma Correction ($\gamma=2.0$)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(s, cmap='gray', vmin=0, vmax=1)
    plt.title(f'Contrast Stretching ($r_1={r1}, r_2={r2}$)')
    plt.axis('off')

    plt.tight_layout()
    
    # Save the combined grid plot BEFORE plt.show()
    grid_save_path = os.path.join(output_folder, 'q1_grid_plot.png')
    plt.savefig(grid_save_path)
    print(f"Grid plot saved to {grid_save_path}")

    # Show the plot window
    plt.show()