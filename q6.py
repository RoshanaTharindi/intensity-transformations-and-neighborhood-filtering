import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# --- Create/Check results folder ---
output_folder = 'results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def generate_dog_kernels(size, sigma):
    """
    Computes the Derivative of Gaussian (DoG) kernels in x and y directions.
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    X, Y = np.meshgrid(ax, ax)
    
    # 1. Base Normalized Gaussian
    G = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    G = G / np.sum(G) # Normalize so it sums to 1
    
    # 2. Apply the mathematical derivative formulas we proved in Part (a)
    Gx = -(X / sigma**2) * G
    Gy = -(Y / sigma**2) * G
    
    return Gx, Gy

# --- Load Image ---
img_path = 'images/einstein.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Could not load any image!")
else:
    # ==========================================
    # (b) Compute 5x5 DoG kernels for sigma = 2
    # ==========================================
    sigma_b = 2.0
    Gx_5x5, Gy_5x5 = generate_dog_kernels(size=5, sigma=sigma_b)
    
    print("-" * 50)
    print("(b): 5x5 Derivative of Gaussian Kernel (X-direction):")
    np.set_printoptions(precision=4, suppress=True)
    print(Gx_5x5)
    print("\n(b): 5x5 Derivative of Gaussian Kernel (Y-direction):")
    print(Gy_5x5)
    print("-" * 50)


    # ==========================================
    # (c) Visualize 51x51 kernel as 3D surface
    # ==========================================
    size_c = 51
    sigma_c = 8.0 # Larger sigma to make the 3D surface visible and smooth
    Gx_51x51, _ = generate_dog_kernels(size=size_c, sigma=sigma_c)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax_range = np.linspace(-(size_c // 2), size_c // 2, size_c)
    X_plot, Y_plot = np.meshgrid(ax_range, ax_range)
    
    # Plotting the X-direction derivative
    surf = ax.plot_surface(X_plot, Y_plot, Gx_51x51, cmap='coolwarm', edgecolor='none')
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.title(f'51x51 Derivative of Gaussian (X-Direction)\n$\sigma={sigma_c}$')
    
    plot_save_path = os.path.join(output_folder, 'q6_3d_dog_x.png')
    plt.savefig(plot_save_path)
    print(f"Saved 3D plot to {plot_save_path}")


    # ==========================================
    # (d) Apply DoG Kernels to Image
    # ==========================================
    # We use cv2.CV_64F to allow negative gradient values during calculation
    grad_x_dog = cv2.filter2D(img, cv2.CV_64F, Gx_5x5)
    grad_y_dog = cv2.filter2D(img, cv2.CV_64F, Gy_5x5)

    # Take absolute value and convert back to 8-bit image for viewing
    abs_grad_x_dog = cv2.convertScaleAbs(grad_x_dog)
    abs_grad_y_dog = cv2.convertScaleAbs(grad_y_dog)


    # ==========================================
    # (e) Compare with OpenCV Sobel
    # ==========================================
    # Sobel with kernel size 5
    grad_x_sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    grad_y_sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    abs_grad_x_sobel = cv2.convertScaleAbs(grad_x_sobel)
    abs_grad_y_sobel = cv2.convertScaleAbs(grad_y_sobel)

    # Save outputs
    cv2.imwrite(os.path.join(output_folder, 'q6_dog_x.jpg'), abs_grad_x_dog)
    cv2.imwrite(os.path.join(output_folder, 'q6_sobel_x.jpg'), abs_grad_x_sobel)

    # --- Plot the Comparisons ---
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(abs_grad_x_dog, cmap='gray')
    plt.title('Derivative of Gaussian (X)')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(abs_grad_x_sobel, cmap='gray')
    plt.title('Sobel Filter (X, ksize=5)')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(abs_grad_y_dog, cmap='gray')
    plt.title('Derivative of Gaussian (Y)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(abs_grad_y_sobel, cmap='gray')
    plt.title('Sobel Filter (Y, ksize=5)')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'q6_comparison_plot.png'))
    plt.show()
