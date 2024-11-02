import cv2
import numpy as np
import glob

def color_normalization(image, target_mean=128, target_std=50):     # Normalize the colors
    # Split the image into its color channels
    channels = cv2.split(image)
    
    # For each channel, normalize with respect to mean and standard deviation
    normalized_channels = []
    for channel in channels:
        # Calculate the mean and standard deviation of the channel
        mean, std = cv2.meanStdDev(channel)
        normalized_channel = (channel - mean[0][0]) / (std[0][0] + 1e-6) * target_std + target_mean
        normalized_channel = np.clip(normalized_channel, 0, 255).astype(np.uint8)
        normalized_channels.append(normalized_channel)

    normalized_image = cv2.merge(normalized_channels)
    
    return normalized_image

# Load images
image1 = cv2.imread('IPCV\SubjectPictures\subject1\subject1Left\subject1_Left_1.jpg')

# Apply global color normalization
normalized_image = color_normalization(image1)

# Save or display the normalized image
cv2.imwrite("IPCV/SubjectPictures/ColorNormalized/subject1/subject1Left/test.jpg", normalized_image)
cv2.imshow('Normalized Image', normalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


def remove_background(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=25, threshold2=65)
    kernel = np.ones((11, 11), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    edges_eroded = cv2.erode(edges_dilated, kernel, iterations=1)
    contours, _ = cv2.findContours(edges_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

# Load an image (replace with the path to your 3D surface mesh image)
image = cv2.imread('IPCV\SubjectPictures\subject1\subject1Left\subject1_Left_1.jpg')

# Remove background
foreground = remove_background(image)

# Save or display the result
cv2.imwrite("IPCV/SubjectPictures/ColorNormalized/subject1/subject1Left/test_bg.jpg", foreground)
cv2.imshow('Foreground', foreground)
cv2.waitKey(0)
cv2.destroyAllWindows()