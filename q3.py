import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Create a folder to save results ---
output_folder = 'results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- Define the Custom Histogram Equalization Function ---
def custom_histogram_equalization(img):
    """
    Equalizes the histogram of a grayscale image from scratch.
    """
    # 1. Calculate the histogram (count of each pixel intensity from 0 to 255)
    hist = np.bincount(img.flatten(), minlength=256)
    
    # 2. Calculate the Probability Density Function (PDF)
    total_pixels = img.size
    pdf = hist / total_pixels
    
    # 3. Calculate the Cumulative Distribution Function (CDF)
    cdf = np.cumsum(pdf)
    
    # 4. Create the mapping transformation (Multiply CDF by 255 and round)
    transform_map = np.round(cdf * 255).astype(np.uint8)
    
    # 5. Apply the mapping to the original image
    equalized_img = transform_map[img]
    
    return equalized_img, hist

# --- 3. Apply the function to the Runway image ---
img_path = 'images/runway.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Could not load {img_path}. Check the filename!")
else:
    # Run our custom function
    eq_img, original_hist = custom_histogram_equalization(img)
    
    # Calculate the histogram of the NEW equalized image for plotting
    eq_hist = np.bincount(eq_img.flatten(), minlength=256)
    
    # Save the standalone equalized image
    cv2.imwrite(os.path.join(output_folder, 'q3_equalized_runway.jpg'), eq_img)
    print(f"Equalized image saved to '{output_folder}' folder.")

    # --- Plot the Results ---
    plt.figure(figsize=(14, 10))
    
    # 1. Original Image
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title('Original Runway Image')
    plt.axis('off')

    # 2. Equalized Image
    plt.subplot(2, 2, 2)
    plt.imshow(eq_img, cmap='gray', vmin=0, vmax=255)
    plt.title('Equalized Image (Custom Function)')
    plt.axis('off')

    # 3. Original Histogram
    plt.subplot(2, 2, 3)
    plt.bar(np.arange(256), original_hist, color='gray', width=1)
    plt.title('Original Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    # 4. Equalized Histogram
    plt.subplot(2, 2, 4)
    plt.bar(np.arange(256), eq_hist, color='gray', width=1)
    plt.title('Equalized Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    plt.tight_layout()
    
    # Save the plot
    grid_save_path = os.path.join(output_folder, 'q3_hist_eq_plot.png')
    plt.savefig(grid_save_path)
    print(f"Plot saved to {grid_save_path}")

    # Show the plot window
    plt.show()