import cv2
from PIL import Image
from PIL.ImageOps import grayscale
from matplotlib import pyplot as plt
import numpy as np

def func1(path):
    # Load image
    image = cv2.imread(path)

    # Display image for 5 seconds
    cv2.imshow("Original image", image)

    cv2.waitKey(5 * 10**3)
    cv2.destroyAllWindows()

# Second part of the homework
def func2(path):
    # Load image
    image = cv2.imread(path)

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # from RGB to Grayscale
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # from RGB to HSV

    # Save Grayscale and HSV images
    cv2.imwrite("grayscale.jpg", grayscale_image)
    cv2.imwrite("hsv.jpg", hsv_image)

    # Display Grayscale and HSV images for 5 seconds
    cv2.imshow("GRAYSCALE", grayscale_image)
    cv2.imshow("HSV", hsv_image)

    cv2.waitKey(5 * 10**3)
    cv2.destroyAllWindows()

    # Brightness histograms for original and Grayscale images
    original_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    grayscale_hist = cv2.calcHist([grayscale_image], [0], None, [256], [0, 256])

    # Plot brightness histograms
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Brightness Histogram Original Image")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.plot(original_hist, color='black')
    plt.xlim([0, 256])

    # Save the histogram
    plt.savefig("brightness_histogram_original.png")

    plt.subplot(1, 2, 2)
    plt.title("Brightness Histogram Grayscale Image")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.plot(grayscale_hist, color='black')
    plt.xlim([0, 256])

    # Save the histogram
    plt.savefig("brightness_histogram_grayscale.png")
    plt.show()

# Third part of the homework
def func3(path):
    # Load image (Grayscale image is assumed)
    image = cv2.imread(path)

    # Define the kernel size used for filters
    kernel_size = 7
    kernel_size_v2 = 3

    # Apply Median filter to Grayscale image
    median_grayscale = cv2.medianBlur(image, kernel_size)
    median_grayscale_v2 = cv2.medianBlur(image, kernel_size_v2)

    # Apply Gaussian smoothing to Grayscale image
    gaussian_grayscale_sigma_v1 = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0.001)
    gaussian_grayscale_sigma_v2 = cv2.GaussianBlur(image, (kernel_size, kernel_size), 5)

    # Apply Laplacian filter for sharpening to Grayscale image
    init_laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=1)
    laplacian_grayscale = np.uint8(np.absolute(init_laplacian))

    # Each filtered image is displayed for 5 seconds
    cv2.imshow("Median Filter applied to Grayscale image with kernel size = 7", median_grayscale)
    cv2.imshow("Median Filter applied to Grayscale image with kernel size = 3", median_grayscale_v2)
    cv2.imshow("Gaussian Smoothing applied to Grayscale image, sigma = 0.001", gaussian_grayscale_sigma_v1)
    cv2.imshow("Gaussian Smoothing applied to Grayscale image, sigma = 5", gaussian_grayscale_sigma_v2)
    cv2.imshow("Laplacian filter applied to Grayscale image", laplacian_grayscale)

    cv2.imwrite("median_grayscale_7.jpg", median_grayscale)
    cv2.imwrite("median_grayscale_3.jpg", median_grayscale_v2)
    cv2.imwrite("gaussian_grayscale_0.001.jpg", gaussian_grayscale_sigma_v1)
    cv2.imwrite("gaussian_grayscale_5.jpg", gaussian_grayscale_sigma_v2)
    cv2.imwrite("laplacian_grayscale.jpg", laplacian_grayscale)

    cv2.waitKey(5 * 10**3)
    cv2.destroyAllWindows()

# Fourth part of the homework
def func4(path):
    # Load image
    image = cv2.imread(path)

    # Apply Sobel operator to image to compute horizontal and vertical gradients
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) # Horizontal gradients
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3) # Vertical gradients

    # Apply the Canny edge detection algorithm
    edges = cv2.Canny(image, 100, 200)

    # Apply the Harris corner detector to identify corner points on Grayscale image
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = np.float32(grayscale_image)
    corners = cv2.cornerHarris(grayscale_image, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)
    corners_original = image.copy()
    corners_original[corners > 0.01 * corners.max()] = [0, 0, 255]

    # Display images for 5 seconds
    cv2.imshow("Sobel X (Horizontal Edges)", sobel_x)
    cv2.imshow("Sobel Y (Vertical Edges)", sobel_y)
    cv2.imshow("Corners", corners_original)
    cv2.imshow("Edges", edges)

    cv2.imwrite("sobel_x.jpg", sobel_x)
    cv2.imwrite("sobel_y.jpg", sobel_y)
    cv2.imwrite("corners.jpg", corners_original)
    cv2.imwrite("edges.jpg", edges)

    cv2.waitKey(5 * 10**3)
    cv2.destroyAllWindows()

# Fifth part of the homework
def func5(path):
    # Load image (Grayscale image is assumed)
    grayscale_image = cv2.imread(path)

    # Define the threshold value
    threshold_value = 200

    # Binarize the Grayscale image using threshold segmentation
    _, binary_grayscale = cv2.threshold(grayscale_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Define the kernel
    kernel = np.ones((3, 3), np.uint8)

    # Apply erosion operation to binary grayscale image
    eroded_binary_grayscale = cv2.erode(binary_grayscale, kernel, iterations=10)

    # Apply dilation operation to binary grayscale image
    dilated_binary_grayscale = cv2.dilate(binary_grayscale, kernel, iterations=5)

    # Display images for 5 seconds
    cv2.imshow("Binary grayscale image", binary_grayscale)
    cv2.imshow("Binary grayscale image after erosion", eroded_binary_grayscale)
    cv2.imshow("Binary grayscale image after dilation", dilated_binary_grayscale)

    cv2.imwrite("binary_grayscale.jpg", binary_grayscale)
    cv2.imwrite("eroded_gray.jpg", eroded_binary_grayscale)
    cv2.imwrite("dilation.jpg", dilated_binary_grayscale)

    cv2.waitKey(5 * 10**3)
    cv2.destroyAllWindows()

# Path to the original image
original_path = "example.jpg"
# func1(original_path)
# func2(original_path)
# func3("grayscale.jpg")
# func4("example.jpg")
# func5("grayscale.jpg")