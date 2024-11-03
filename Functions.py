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
            print(f"Undistorted image saved to {output_path}")

def color_normalization_folder(input_folder, output_folder, target_mean=128, target_std=75):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Read the image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Failed to load image {filename}")
                continue
            
            # Split image into color channels
            channels = cv2.split(image)
            
            # Normalize each channel
            normalized_channels = []
            for channel in channels:
                mean, std = cv2.meanStdDev(channel)
                normalized_channel = (channel - mean[0][0]) / (std[0][0] + 1e-6) * target_std + target_mean
                normalized_channel = np.clip(normalized_channel, 0, 255).astype(np.uint8)
                normalized_channels.append(normalized_channel)
            
            # Merge normalized channels back into an image
            normalized_image = cv2.merge(normalized_channels)
            
            # Save the normalized image to the output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, normalized_image)
            print(f"Normalized image saved to {output_path}")

# def remove_background_folder(input_folder, output_folder):
#     # Create the output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Loop through all files in the input folder
#     for filename in os.listdir(input_folder):
#         input_path = os.path.join(input_folder, filename)
        
#         # Check if the file is an image
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
#             # Read the image
#             image = cv2.imread(input_path)
#             if image is None:
#                 print(f"Failed to load image {filename}")
#                 continue
            
#             # Convert image to grayscale
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
#             # Use edge detection
#             edges = cv2.Canny(gray, threshold1=20, threshold2=75)
#             kernel = np.ones((11, 11), np.uint8)
#             edges_dilated = cv2.dilate(edges, kernel, iterations=2)
#             edges_eroded = cv2.erode(edges_dilated, kernel, iterations=1)
            
#             # Find contours
#             contours, _ = cv2.findContours(edges_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             mask = np.zeros_like(gray)
#             cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
            
#             # Apply mask to the original image
#             result = cv2.bitwise_and(image, image, mask=mask)
            
#             # Save the result to the output folder
#             output_path = os.path.join(output_folder, filename)
#             cv2.imwrite(output_path, result)
#             print(f"Processed image saved to {output_path}")


def remove_background_folder(input_folder, output_folder, Sobel1, Sobel2):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Read the image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Failed to load image {filename}")
                continue
            
            # Convert image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


            # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #     sobelx = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)  # Sobel in x-direction
            #     sobely = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)  # Sobel in y-direction
            #     sobel_magnitude = cv2.magnitude(sobelx, sobely)
            #     sobel_frame = np.uint8(np.absolute(sobel_magnitude))

            # Apply Sobel edge detection
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x direction
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y direction
            
            # Combine the gradients
            edges = np.uint8(np.absolute(cv2.magnitude(sobel_x, sobel_y)))
            edges = np.uint8(np.clip(edges, 0, 255))  # Convert to uint8

            # Threshold the edges to get a binary image
            _, edges_binary = cv2.threshold(edges, Sobel1, Sobel2, cv2.THRESH_BINARY)

            # Apply morphological operations to remove noise
            kernel = np.ones((3, 3), np.uint8)
            edges_dilated = cv2.dilate(edges_binary, kernel, iterations=1)
            edges_eroded = cv2.erode(edges_dilated, kernel, iterations=1)

            # Find contours
            contours, _ = cv2.findContours(edges_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(gray)

            # Draw the largest contour filled
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

            # Apply the mask to the original image
            result = cv2.bitwise_and(image, image, mask=mask)
            
            # Save the result to the output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, result)
            print(f"Processed image saved to {output_path}")


# # Subject1 Left
# intrinsicMatrixL, distortionMatrixL = calibrate_camera_intrinsic(9, 6, "IPCV/CalibrationPictures/Calibratie 1/calibrationLeft/*.jpg")
# undistort_images_in_folder("IPCV/SubjectPictures/subject1/subject1Left", "IPCV/SubjectPicturesUndistorted/subject1/subject1Left", intrinsicMatrixL, distortionMatrixL)
# color_normalization_folder("IPCV/SubjectPicturesUndistorted/subject1/subject1Left", "IPCV/SubjectPicturesNormalized/subject1/subject1Left")
# remove_background_folder("IPCV/SubjectPicturesNormalized/subject1/subject1Left", "IPCV/SubjectPicturesProcessed/subject1/subject1Left", 20, 150)

# # Subject1 Middle
# intrinsicMatrixM, distortionMatrixM = calibrate_camera_intrinsic(9, 6, "IPCV/CalibrationPictures/Calibratie 1/calibrationMiddle/*.jpg")
# undistort_images_in_folder("IPCV/SubjectPictures/subject1/subject1Middle", "IPCV/SubjectPicturesUndistorted/subject1/subject1Middle", intrinsicMatrixM, distortionMatrixM)
# color_normalization_folder("IPCV/SubjectPicturesUndistorted/subject1/subject1Middle", "IPCV/SubjectPicturesNormalized/subject1/subject1Middle")
# remove_background_folder("IPCV/SubjectPicturesNormalized/subject1/subject1Middle", "IPCV/SubjectPicturesProcessed/subject1/subject1Middle", 20, 150)

# # Subject1 Right
# intrinsicMatrixR, distortionMatrixR = calibrate_camera_intrinsic(9, 6, "IPCV/CalibrationPictures/Calibratie 1/calibrationRight/*.jpg")
# undistort_images_in_folder("IPCV/SubjectPictures/subject1/subject1Right", "IPCV/SubjectPicturesUndistorted/subject1/subject1Right", intrinsicMatrixR, distortionMatrixR)
# color_normalization_folder("IPCV/SubjectPicturesUndistorted/subject1/subject1Right", "IPCV/SubjectPicturesNormalized/subject1/subject1Right")
# remove_background_folder("IPCV/SubjectPicturesNormalized/subject1/subject1Right", "IPCV/SubjectPicturesProcessed/subject1/subject1Right", 20, 150)

# # Subject 2 Left
# intrinsicMatrixL2, distortionMatrixL2 = calibrate_camera_intrinsic(9, 6, "IPCV/CalibrationPictures/Calibratie 1/calibrationLeft/*.jpg")
# undistort_images_in_folder("IPCV/SubjectPictures/subject2/subject2_Left", "IPCV/SubjectPicturesUndistorted/subject2/subject2Left", intrinsicMatrixL, distortionMatrixL)
# color_normalization_folder("IPCV/SubjectPicturesUndistorted/subject2/subject2Left", "IPCV/SubjectPicturesNormalized/subject2/subject2Left")
# remove_background_folder("IPCV/SubjectPicturesNormalized/subject2/subject2Left", "IPCV/SubjectPicturesProcessed/subject2/subject2Left", 15, 200)

# # Subject 2 Middle
# intrinsicMatrixM2, distortionMatrixM2 = calibrate_camera_intrinsic(9, 6, "IPCV/CalibrationPictures/Calibratie 1/calibrationMiddle/*.jpg")
# undistort_images_in_folder("IPCV/SubjectPictures/subject2/subject2_Middle", "IPCV/SubjectPicturesUndistorted/subject2/subject2Middle", intrinsicMatrixM, distortionMatrixM)
# color_normalization_folder("IPCV/SubjectPicturesUndistorted/subject2/subject2Middle", "IPCV/SubjectPicturesNormalized/subject2/subject2Middle")
# remove_background_folder("IPCV/SubjectPicturesNormalized/subject2/subject2Middle", "IPCV/SubjectPicturesProcessed/subject2/subject2Middle", 17, 20)

# # Subject 2 Right
# intrinsicMatrixR2, distortionMatrixR2 = calibrate_camera_intrinsic(9, 6, "IPCV/CalibrationPictures/Calibratie 1/calibrationRight/*.jpg")
# undistort_images_in_folder("IPCV/SubjectPictures/subject2/subject2_Right", "IPCV/SubjectPicturesUndistorted/subject2/subject2Right", intrinsicMatrixR, distortionMatrixR)
# color_normalization_folder("IPCV/SubjectPicturesUndistorted/subject2/subject2Right", "IPCV/SubjectPicturesNormalized/subject2/subject2Right")
# remove_background_folder("IPCV/SubjectPicturesNormalized/subject2/subject2Right", "IPCV/SubjectPicturesProcessed/subject2/subject2Right", 15, 200)

# # Subject 4 Left
# intrinsicMatrixL4, distortionMatrixL4 = calibrate_camera_intrinsic(9, 6, "IPCV/CalibrationPictures/Calibratie 1/calibrationLeft/*.jpg")
# undistort_images_in_folder("IPCV/SubjectPictures/subject4/subject4_Left", "IPCV/SubjectPicturesUndistorted/subject4/subject4Left", intrinsicMatrixL, distortionMatrixL)
# color_normalization_folder("IPCV/SubjectPicturesUndistorted/subject4/subject4Left", "IPCV/SubjectPicturesNormalized/subject4/subject4Left")
# remove_background_folder("IPCV/SubjectPicturesNormalized/subject4/subject4Left", "IPCV/SubjectPicturesProcessed/subject4/subject4Left", 20, 150)

# # Subject 4 Middle
# intrinsicMatrixM4, distortionMatrixM4 = calibrate_camera_intrinsic(9, 6, "IPCV/CalibrationPictures/Calibratie 1/calibrationMiddle/*.jpg")
# undistort_images_in_folder("IPCV/SubjectPictures/subject4/subject4_Middle", "IPCV/SubjectPicturesUndistorted/subject4/subject4Middle", intrinsicMatrixM, distortionMatrixM)
# color_normalization_folder("IPCV/SubjectPicturesUndistorted/subject4/subject4Middle", "IPCV/SubjectPicturesNormalized/subject4/subject4Middle")
# remove_background_folder("IPCV/SubjectPicturesNormalized/subject4/subject4Middle", "IPCV/SubjectPicturesProcessed/subject4/subject4Middle", 20, 150)

# # Subject 4 Right
# intrinsicMatrixR4, distortionMatrixR4 = calibrate_camera_intrinsic(9, 6, "IPCV/CalibrationPictures/Calibratie 1/calibrationRight/*.jpg")
# undistort_images_in_folder("IPCV/SubjectPictures/subject4/subject4_Right", "IPCV/SubjectPicturesUndistorted/subject4/subject4Right", intrinsicMatrixR, distortionMatrixR)
# color_normalization_folder("IPCV/SubjectPicturesUndistorted/subject4/subject4Right", "IPCV/SubjectPicturesNormalized/subject4/subject4Right")
# remove_background_folder("IPCV/SubjectPicturesNormalized/subject4/subject4Right", "IPCV/SubjectPicturesProcessed/subject4/subject4Right", 20, 150)



# Subject 1 Left
intrinsicMatrixL, distortionMatrixL = calibrate_camera_intrinsic(9, 6, "IPCV/CalibrationPictures/Calibratie 1/calibrationLeft/*.jpg")
undistort_images_in_folder("IPCV/SubjectPictures/subject1/subject1Left", "IPCV/SubjectPicturesUndistorted/subject1/subject1Left", intrinsicMatrixL, distortionMatrixL)
remove_background_folder("IPCV/SubjectPicturesUndistorted/subject1/subject1Left", "IPCV/SubjectPicturesProcessed/subject1/subject1Left", 20, 150)
color_normalization_folder("IPCV/SubjectPicturesProcessed/subject1/subject1Left", "IPCV/SubjectPicturesNormalized/subject1/subject1Left")

# Subject 1 Middle
intrinsicMatrixM, distortionMatrixM = calibrate_camera_intrinsic(9, 6, "IPCV/CalibrationPictures/Calibratie 1/calibrationMiddle/*.jpg")
undistort_images_in_folder("IPCV/SubjectPictures/subject1/subject1Middle", "IPCV/SubjectPicturesUndistorted/subject1/subject1Middle", intrinsicMatrixM, distortionMatrixM)
remove_background_folder("IPCV/SubjectPicturesUndistorted/subject1/subject1Middle", "IPCV/SubjectPicturesProcessed/subject1/subject1Middle", 20, 150)
color_normalization_folder("IPCV/SubjectPicturesProcessed/subject1/subject1Middle", "IPCV/SubjectPicturesNormalized/subject1/subject1Middle")

# Subject 1 Right
intrinsicMatrixR, distortionMatrixR = calibrate_camera_intrinsic(9, 6, "IPCV/CalibrationPictures/Calibratie 1/calibrationRight/*.jpg")
undistort_images_in_folder("IPCV/SubjectPictures/subject1/subject1Right", "IPCV/SubjectPicturesUndistorted/subject1/subject1Right", intrinsicMatrixR, distortionMatrixR)
remove_background_folder("IPCV/SubjectPicturesUndistorted/subject1/subject1Right", "IPCV/SubjectPicturesProcessed/subject1/subject1Right", 10, 150)
color_normalization_folder("IPCV/SubjectPicturesProcessed/subject1/subject1Right", "IPCV/SubjectPicturesNormalized/subject1/subject1Right")

# Subject 2 Left
intrinsicMatrixL2, distortionMatrixL2 = calibrate_camera_intrinsic(9, 6, "IPCV/CalibrationPictures/Calibratie 1/calibrationLeft/*.jpg")
undistort_images_in_folder("IPCV/SubjectPictures/subject2/subject2_Left", "IPCV/SubjectPicturesUndistorted/subject2/subject2Left", intrinsicMatrixL2, distortionMatrixL2)
remove_background_folder("IPCV/SubjectPicturesUndistorted/subject2/subject2Left", "IPCV/SubjectPicturesProcessed/subject2/subject2Left", 6, 75)
color_normalization_folder("IPCV/SubjectPicturesProcessed/subject2/subject2Left", "IPCV/SubjectPicturesNormalized/subject2/subject2Left")

# Subject 2 Middle
intrinsicMatrixM2, distortionMatrixM2 = calibrate_camera_intrinsic(9, 6, "IPCV/CalibrationPictures/Calibratie 1/calibrationMiddle/*.jpg")
undistort_images_in_folder("IPCV/SubjectPictures/subject2/subject2_Middle", "IPCV/SubjectPicturesUndistorted/subject2/subject2Middle", intrinsicMatrixM2, distortionMatrixM2)
remove_background_folder("IPCV/SubjectPicturesUndistorted/subject2/subject2Middle", "IPCV/SubjectPicturesProcessed/subject2/subject2Middle", 6, 75)
color_normalization_folder("IPCV/SubjectPicturesProcessed/subject2/subject2Middle", "IPCV/SubjectPicturesNormalized/subject2/subject2Middle")

# Subject 2 Right
intrinsicMatrixR2, distortionMatrixR2 = calibrate_camera_intrinsic(9, 6, "IPCV/CalibrationPictures/Calibratie 1/calibrationRight/*.jpg")
undistort_images_in_folder("IPCV/SubjectPictures/subject2/subject2_Right", "IPCV/SubjectPicturesUndistorted/subject2/subject2Right", intrinsicMatrixR2, distortionMatrixR2)
remove_background_folder("IPCV/SubjectPicturesUndistorted/subject2/subject2Right", "IPCV/SubjectPicturesProcessed/subject2/subject2Right", 6, 75)
color_normalization_folder("IPCV/SubjectPicturesProcessed/subject2/subject2Right", "IPCV/SubjectPicturesNormalized/subject2/subject2Right")

# Subject 4 Left
intrinsicMatrixL4, distortionMatrixL4 = calibrate_camera_intrinsic(9, 6, "IPCV/CalibrationPictures/Calibratie 1/calibrationLeft/*.jpg")
undistort_images_in_folder("IPCV/SubjectPictures/subject4/subject4_Left", "IPCV/SubjectPicturesUndistorted/subject4/subject4Left", intrinsicMatrixL4, distortionMatrixL4)
remove_background_folder("IPCV/SubjectPicturesUndistorted/subject4/subject4Left", "IPCV/SubjectPicturesProcessed/subject4/subject4Left", 6, 12)
color_normalization_folder("IPCV/SubjectPicturesProcessed/subject4/subject4Left", "IPCV/SubjectPicturesNormalized/subject4/subject4Left")

# Subject 4 Middle
intrinsicMatrixM4, distortionMatrixM4 = calibrate_camera_intrinsic(9, 6, "IPCV/CalibrationPictures/Calibratie 1/calibrationMiddle/*.jpg")
undistort_images_in_folder("IPCV/SubjectPictures/subject4/subject4_Middle", "IPCV/SubjectPicturesUndistorted/subject4/subject4Middle", intrinsicMatrixM4, distortionMatrixM4)
remove_background_folder("IPCV/SubjectPicturesUndistorted/subject4/subject4Middle", "IPCV/SubjectPicturesProcessed/subject4/subject4Middle", 6, 12)
color_normalization_folder("IPCV/SubjectPicturesProcessed/subject4/subject4Middle", "IPCV/SubjectPicturesNormalized/subject4/subject4Middle")

# Subject 4 Right
intrinsicMatrixR4, distortionMatrixR4 = calibrate_camera_intrinsic(9, 6, "IPCV/CalibrationPictures/Calibratie 1/calibrationRight/*.jpg")
undistort_images_in_folder("IPCV/SubjectPictures/subject4/subject4_Right", "IPCV/SubjectPicturesUndistorted/subject4/subject4Right", intrinsicMatrixR4, distortionMatrixR4)
remove_background_folder("IPCV/SubjectPicturesUndistorted/subject4/subject4Right", "IPCV/SubjectPicturesProcessed/subject4/subject4Right", 6, 12)
color_normalization_folder("IPCV/SubjectPicturesProcessed/subject4/subject4Right", "IPCV/SubjectPicturesNormalized/subject4/subject4Right")
