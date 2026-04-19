import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Create/Check results folder ---
output_folder = 'results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- Load Image ---
img_path = 'images/a1q8images/taylor.jpg'
img_bgr = cv2.imread(img_path)

if img_bgr is None:
    print(f"Error: Could not load '{img_path}'. Please check the filename!")
else:
    # Convert BGR (OpenCV default) to RGB for correct colors in matplotlib
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ==========================================
    # Apply Image Sharpening
    # ==========================================

    sharpening_kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])

    # Apply the standard filter
    sharpened_img = cv2.filter2D(img_rgb, -1, sharpening_kernel)
    
    # Create an "Intense" sharpening kernel for comparison
    strong_kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])
    strong_sharpened_img = cv2.filter2D(img_rgb, -1, strong_kernel)

    # --- Save the results ---
    cv2.imwrite(os.path.join(output_folder, 'q9_sharpened.jpg'), cv2.cvtColor(sharpened_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_folder, 'q9_sharpened_strong.jpg'), cv2.cvtColor(strong_sharpened_img, cv2.COLOR_RGB2BGR))
    print(f"Sharpened images saved to '{output_folder}' folder.")

    # --- Plot the Results ---
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(sharpened_img)
    plt.title('Sharpened (Standard Kernel)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(strong_sharpened_img)
    plt.title('Sharpened (Strong Kernel)')
    plt.axis('off')

    plt.tight_layout()
    plot_save_path = os.path.join(output_folder, 'q9_sharpening_comparison.png')
    plt.savefig(plot_save_path)

    # Show the plot window
    plt.show()