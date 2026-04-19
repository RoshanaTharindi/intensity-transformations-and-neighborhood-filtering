import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Create/Check results folder ---
output_folder = 'results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- Load Image ---
img_path = 'images/emma.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Could not load '{img_path}'. Please check the filename!")
else:
    # ==========================================
    # (a) Apply Gaussian smoothing
    # ==========================================
    # We use a 5x5 kernel. Gaussian blur averages the surrounding pixels.
    gaussian_smoothed = cv2.GaussianBlur(img, (5, 5), 0)

    # ==========================================
    # (b) Apply Median filtering
    # ==========================================
    # We use a 5x5 kernel size. Median filter picks the middle value of the sorted neighborhood.
    median_smoothed = cv2.medianBlur(img, 5)

    # --- Save the results ---
    cv2.imwrite(os.path.join(output_folder, 'q8_gaussian_smoothed.jpg'), gaussian_smoothed)
    cv2.imwrite(os.path.join(output_folder, 'q8_median_smoothed.jpg'), median_smoothed)
    print(f"Processed images saved to '{output_folder}' folder.")

    # --- Plot the Results for Comparison ---
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original (Salt & Pepper Noise)')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(gaussian_smoothed, cmap='gray')
    plt.title('Gaussian Smoothing (5x5)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(median_smoothed, cmap='gray')
    plt.title('Median Filtering (5x5)')
    plt.axis('off')

    plt.tight_layout()
    plot_save_path = os.path.join(output_folder, 'q8_noise_comparison.png')
    plt.savefig(plot_save_path)
 
    # Show the plot window
    plt.show()