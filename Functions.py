import cv2
import numpy as np
import glob
import os

def calibrate_camera_intrinsic(checkerboard_dim1, checkerboard_dim2, image_folder):
    checkerboard_dims = (checkerboard_dim1, checkerboard_dim2)

    # Criteria for refining the corners for better accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points based on checkerboard dimensions (e.g., (0,0,0), (1,0,0), ..., (8,5,0))
    objp = np.zeros((checkerboard_dims[0] * checkerboard_dims[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)
    

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    images = sorted(glob.glob(image_folder))
    image_shape = None

    # Iterate over images to find chessboard corners
    for img_path in images:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Capture the image size from the first image
        if image_shape is None:
            image_shape = gray.shape[::-1]
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_dims, None)
        
        # If corners are found, refine them and add to the lists
        if ret:
            objpoints.append(objp)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_refined)

    # Perform camera calibration to get the intrinsic matrix
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)
    
    return mtx, dist

def undistort_images_in_folder(input_folder, output_folder, intrinsicMatrix, distortionMatrix):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop over each file in the input folder
    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        
        # Check if the file is an image (e.g., .jpg, .png)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to load image {filename}")
                continue
            
            # Get dimensions of the image
            h, w = img.shape[:2]
            
            # Get optimal new camera matrix
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsicMatrix, distortionMatrix, (w,h), 1, (w,h))
            
            # Undistort the image
            dst = cv2.undistort(img, intrinsicMatrix, distortionMatrix, None, newcameramtx)
            
            # Crop the image based on the region of interest
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            
            # Save the undistorted image to the output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, dst)
            print(f"Saved undistorted image: {output_path}")

def color_normalization(image, target_mean=128, target_std=50):     # Normalize the colors
    # Split image in channels
    channels = cv2.split(image)
    
    # Normalize for each channel
    normalized_channels = []
    for channel in channels:
        mean, std = cv2.meanStdDev(channel)
        normalized_channel = (channel - mean[0][0]) / (std[0][0] + 1e-6) * target_std + target_mean
        normalized_channel = np.clip(normalized_channel, 0, 255).astype(np.uint8)
        normalized_channels.append(normalized_channel)

    # Merge normalized channels
    normalized_image = cv2.merge(normalized_channels)
    
    return normalized_image

def remove_background(image):
    # Use edge detection
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


# Get intrinsic matrix
intrinsicMatrix, distortionMatrix = calibrate_camera_intrinsic(9, 6, "IPCV/CalibrationPictures/Calibratie 1/calibrationRight/*.jpg")
print(intrinsicMatrix)
print(distortionMatrix)

undistort_images_in_folder("IPCV/SubjectPictures/subject1/subject1Left", "IPCV/Test", intrinsicMatrix, distortionMatrix)
