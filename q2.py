import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

output_folder = 'results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- 2. LAB Color Space Gamma Correction ---
img_path_q2 = 'images/highlights_and_shadows.jpg'
img_bgr = cv2.imread(img_path_q2)

if img_bgr is None:
    print(f"Error: Could not load {img_path_q2}. Check the filename!")
else:
    # OpenCV loads images in BGR format. Convert to RGB for correct matplotlib plotting.
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert the BGR image to LAB color space
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    # Split the LAB image into L, a, and b channels
    l_channel, a_channel, b_channel = cv2.split(img_lab)

    # --- (a) Apply gamma correction to the L plane ---
    # Normalize the L channel to [0, 1] for the power function
    l_normalized = l_channel.astype(np.float32) / 255.0

    # The image has very dark shadows. A gamma < 1 will brighten the dark regions.
    # Let's use gamma = 0.5. (Make sure to state this in your PDF report!)
    gamma_value = 0.5
    l_gamma_corrected = np.power(l_normalized, gamma_value)

    # Scale back to [0, 255] and convert to 8-bit integer
    l_corrected = np.uint8(l_gamma_corrected * 255)

    # Merge the corrected L channel back with the original 'a' and 'b' channels
    img_lab_corrected = cv2.merge((l_corrected, a_channel, b_channel))

    # Convert the corrected LAB image back to BGR (for saving) and RGB (for plotting)
    img_bgr_corrected = cv2.cvtColor(img_lab_corrected, cv2.COLOR_LAB2BGR)
    img_rgb_corrected = cv2.cvtColor(img_lab_corrected, cv2.COLOR_LAB2RGB)
    
    # Save the corrected image standalone
    cv2.imwrite(os.path.join(output_folder, 'q2_corrected_image.jpg'), img_bgr_corrected)
    print(f"Corrected image saved to '{output_folder}' folder.")

    # --- (b) Show the histograms of the original and corrected images ---
    plt.figure(figsize=(14, 10))

    # 1. Original Image
    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    # 2. Corrected Image
    plt.subplot(2, 2, 2)
    plt.imshow(img_rgb_corrected)
    plt.title(f'Corrected Image ($\gamma = {gamma_value}$)')
    plt.axis('off')

    # 3. Histogram of Original L Plane
    plt.subplot(2, 2, 3)
    plt.hist(l_channel.ravel(), bins=256, range=[0, 256], color='gray')
    plt.title('Histogram of Original L Plane')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    # 4. Histogram of Corrected L Plane
    plt.subplot(2, 2, 4)
    plt.hist(l_corrected.ravel(), bins=256, range=[0, 256], color='gray')
    plt.title(f'Histogram of Corrected L Plane ($\gamma = {gamma_value}$)')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    plt.tight_layout()
    
    # Save the plot grid before showing
    grid_save_path = os.path.join(output_folder, 'q2_histograms_plot.png')
    plt.savefig(grid_save_path)
    print(f"Histogram plot saved to {grid_save_path}")

    # Show the plot window
    plt.show()