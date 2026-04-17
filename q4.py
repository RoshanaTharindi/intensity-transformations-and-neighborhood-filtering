import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Create/Check results folder ---
output_folder = 'results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def masked_histogram_equalization(img, mask):
    """
    Applies histogram equalization ONLY to the regions of the image where mask == 255.
    """
    # 1. Extract only the foreground pixels using the mask
    foreground_pixels = img[mask == 255]
    
    # 2. Calculate the histogram of ONLY the foreground pixels
    hist = np.bincount(foreground_pixels, minlength=256)
    
    # 3. Calculate PDF and CDF
    pdf = hist / len(foreground_pixels)
    cdf = np.cumsum(pdf)
    
    # 4. Create the mapping transformation (Multiply CDF by 255 and round)
    transform_map = np.round(cdf * 255).astype(np.uint8)
    
    # 5. Apply the mapping to a copy of the original image
    equalized_img = img.copy()
    
    # Replace the masked pixels with their new equalized values
    equalized_img[mask == 255] = transform_map[img[mask == 255]]
    
    return equalized_img

# --- Load the Image ---
img_path = 'images/woman_standing_in_front_of_an_open_door.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Could not load {img_path}. Check the filename!")
else:
    # --- (a) Otsu Thresholding ---
    thresh_val, mask_inv = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    print(f"(a)The calculated Otsu threshold value is: {thresh_val}")

    # --- (b) Histogram Equalization for the Foreground ---
    # Apply our custom masked equalization function
    eq_img = masked_histogram_equalization(img, mask_inv)
    

    # --- Save the results ---
    cv2.imwrite(os.path.join(output_folder, 'q4_otsu_mask.jpg'), mask_inv)
    cv2.imwrite(os.path.join(output_folder, 'q4_masked_equalization.jpg'), eq_img)
    print(f"Images saved to '{output_folder}' folder.")

    # --- Plot the Results ---
    plt.figure(figsize=(15, 5))
    
    # 1. Original Grayscale
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title('Original Grayscale Image')
    plt.axis('off')

    # 2. Otsu Mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask_inv, cmap='gray', vmin=0, vmax=255)
    plt.title(f'Otsu Mask (Foreground)\nThreshold: {thresh_val}')
    plt.axis('off')

    # 3. Equalized Foreground
    plt.subplot(1, 3, 3)
    plt.imshow(eq_img, cmap='gray', vmin=0, vmax=255)
    plt.title('Masked Histogram Equalization\n(Background Preserved)')
    plt.axis('off')

    plt.tight_layout()
    
    # Save the plot
    grid_save_path = os.path.join(output_folder, 'q4_otsu_eq_plot.png')
    plt.savefig(grid_save_path)
    
    # Show the plot window
    plt.show()