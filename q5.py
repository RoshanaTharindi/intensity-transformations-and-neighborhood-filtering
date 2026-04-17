import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D surface plotting
import os

# --- Create/Check results folder ---
output_folder = 'results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def generate_gaussian_kernel(size, sigma):
    """
    Generates a normalized 2D Gaussian kernel.
    """
    # Create an array of coordinates centered at 0
    # For size 5, this creates: [-2, -1, 0, 1, 2]
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    
    # Calculate the Gaussian function
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    
    # Normalize the kernel so the sum of all elements equals 1
    # This prevents the image from becoming artificially bright or dark
    normalized_kernel = kernel / np.sum(kernel)
    
    return normalized_kernel

# --- Load a Grayscale Image ---
img_path = 'images/einstein.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Could not load any image!")
else:
    # ==========================================
    # (a) Compute normalized 5x5 Gaussian kernel 
    # ==========================================
    sigma_a = 2.0
    kernel_5x5 = generate_gaussian_kernel(size=5, sigma=sigma_a)
    
    print("-" * 50)
    print("(a): Normalized 5x5 Gaussian kernel (sigma=2):")
    np.set_printoptions(precision=4, suppress=True)
    print(kernel_5x5)
    print("-" * 50)


    # ==========================================
    # (b) Visualize 51x51 kernel as 3D surface
    # ==========================================
    size_b = 51
    # We use a larger sigma here so the "bell curve" stretches out and looks nice in 3D
    sigma_b = 8.0 
    kernel_51x51 = generate_gaussian_kernel(size=size_b, sigma=sigma_b)

    # Setup the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create X and Y coordinates for the 3D plot
    ax_range = np.linspace(-(size_b // 2), size_b // 2, size_b)
    X, Y = np.meshgrid(ax_range, ax_range)
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, kernel_51x51, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.title(f'51x51 Gaussian Kernel 3D Surface Plot ($\ sigma={sigma_b}$)')
    
    # Save the 3D plot
    plot_save_path = os.path.join(output_folder, 'q5_3d_gaussian.png')
    plt.savefig(plot_save_path)
    print(f"Saved 3D plot to {plot_save_path}")


    # ==========================================
    # (c) Apply manual kernel (cv2.filter2D)
    # ==========================================
    # The '-1' means the output image will have the same depth as the source
    manual_blur = cv2.filter2D(img, -1, kernel_5x5)


    # ==========================================
    # (d) Apply OpenCV built-in function
    # ==========================================
    # cv2.GaussianBlur takes (image, kernel_size_tuple, sigmaX)
    cv2_blur = cv2.GaussianBlur(img, (5, 5), sigmaX=sigma_a)


    # --- Calculate the Difference ---
    diff = cv2.absdiff(manual_blur, cv2_blur)

    # Save images
    cv2.imwrite(os.path.join(output_folder, 'q5_manual_blur.jpg'), manual_blur)
    cv2.imwrite(os.path.join(output_folder, 'q5_opencv_blur.jpg'), cv2_blur)

    # --- Plot the Results for c and d ---
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(manual_blur, cmap='gray')
    plt.title('Manual 5x5 Gaussian Blur\n(cv2.filter2D)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2_blur, cmap='gray')
    plt.title("OpenCV's cv2.GaussianBlur")
    plt.axis('off')

    plt.tight_layout()
    
    # Save the comparison grid
    grid_save_path = os.path.join(output_folder, 'q5_blur_comparison.png')
    plt.savefig(grid_save_path)
    
    plt.show()