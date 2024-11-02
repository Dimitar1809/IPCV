import cv2
import numpy as np
import glob
import open3d as o3d


CHECKERBOARD = (9, 6)
SQUARE_SIZE = 0.01  # Square size in meters (10 mm)

image_shape = None  

images_left = sorted(glob.glob('CalibrationPictures/Calibratie 1/calibrationLeft/*.jpg'))
images_middle = sorted(glob.glob('CalibrationPictures/Calibratie 1/calibrationMiddle/*.jpg'))
images_right = sorted(glob.glob('CalibrationPictures/Calibratie 1/calibrationRight/*.jpg'))

# Criteria for termination of the iterative algorithm. Either after 30 iterations or if the change in result is smaller than 0.001 (required accuracy epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D points in real world space (the chessboard corners in a 3D space)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE  # Scale by square size

objpoints = []
imgpoints_left = []
imgpoints_middle = []
imgpoints_right = []

for img_left, img_middle, img_right in zip(images_left, images_middle, images_right):
    left_img = cv2.imread(img_left)
    middle_img = cv2.imread(img_middle)
    right_img = cv2.imread(img_right)

    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_middle = cv2.cvtColor(middle_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    if image_shape is None:
        image_shape = gray_left.shape[::-1]  # Capture only once, assumed all images are the same size

    # Find the chessboard corners in each image
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHECKERBOARD, None)
    ret_middle, corners_middle = cv2.findChessboardCorners(gray_middle, CHECKERBOARD, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHECKERBOARD, None)

    if ret_left and ret_middle and ret_right:
        objpoints.append(objp)

        # Refine the corner positions
        corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        imgpoints_left.append(corners_left)
        corners_middle = cv2.cornerSubPix(gray_middle, corners_middle, (11, 11), (-1, -1), criteria)
        imgpoints_middle.append(corners_middle)
        corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
        imgpoints_right.append(corners_right)

cv2.destroyAllWindows()

# Perform camera calibration for each camera
print("Calibrating Left Camera...")
ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints, imgpoints_left, image_shape, None, None)
print("Calibrating Middle Camera...")
ret_middle, mtx_middle, dist_middle, rvecs_middle, tvecs_middle = cv2.calibrateCamera(objpoints, imgpoints_middle, image_shape, None, None)
print("Calibrating Right Camera...")
ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints, imgpoints_right, image_shape, None, None)

# Stereo Calibration for each pair
flags = cv2.CALIB_FIX_INTRINSIC

# Stereo Calibration between Left and Middle
print("Stereo Calibration between Left and Middle Cameras...")
ret_left_middle, _, _, _, _, R_left_middle, T_left_middle, E_left_middle, F_left_middle = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_middle, mtx_left, dist_left, mtx_middle, dist_middle, image_shape, flags=flags)

# Stereo Calibration between Middle and Right
print("Stereo Calibration between Middle and Right Cameras...")
ret_middle_right, _, _, _, _, R_middle_right, T_middle_right, E_middle_right, F_middle_right = cv2.stereoCalibrate(
    objpoints, imgpoints_middle, imgpoints_right, mtx_middle, dist_middle, mtx_right, dist_right, image_shape, flags=flags)

# Stereo Calibration between Left and Right
print("Stereo Calibration between Left and Right Cameras...")
ret_left_right, _, _, _, _, R_left_right, T_left_right, E_left_right, F_left_right = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right, mtx_left, dist_left, mtx_right, dist_right, image_shape, flags=flags)

# Calculate the baseline between Left and Middle (magnitude of translation vector T_left_middle)
baseline_left_middle = np.linalg.norm(T_left_middle)
print(f"Baseline (distance between Left and Middle): {baseline_left_middle} meters")

# Calculate the baseline between Middle and Right (magnitude of translation vector T_middle_right)
baseline_middle_right = np.linalg.norm(T_middle_right)
print(f"Baseline (distance between Middle and Right): {baseline_middle_right} meters")



# Rectify the stereo images (first left and middle)
# Load the pair of stereo images 
img_left = cv2.imread('SubjectPictures/subject1/subject1Left/subject1_Left_1.jpg')
img_right = cv2.imread('SubjectPictures/subject1/subject1Middle/subject1_Middle_1.jpg')


# Get the image size (assuming both images are the same size)
image_shape = img_left.shape[:2][::-1]  # (width, height)

# Step 1: Stereo Rectification
# Compute the rectification transforms
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    mtx_left, dist_left, mtx_right, dist_right, image_shape, R_left_middle, T_left_middle, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-0.5
)

# Step 2: Compute Rectification Maps
# Compute the rectification maps for each camera
map1_left, map2_left = cv2.initUndistortRectifyMap(mtx_left, dist_left, R1, P1, image_shape, cv2.CV_16SC2)
map1_right, map2_right = cv2.initUndistortRectifyMap(mtx_right, dist_right, R2, P2, image_shape, cv2.CV_16SC2)

# Step 3: Apply Rectification Maps
# Remap the images to their rectified versions
rectified_left = cv2.remap(img_left, map1_left, map2_left, cv2.INTER_LINEAR)
rectified_middle = cv2.remap(img_right, map1_right, map2_right, cv2.INTER_LINEAR)

# Step 4: Display the Rectified Images
# Display them side by side to see if the rows are aligned
cv2.imshow('Rectified Left', rectified_left)
cv2.imshow('Rectified M', rectified_middle)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the rectified images for further processing
cv2.imwrite('rectified_left.jpg', rectified_left)
cv2.imwrite('rectified_middle.jpg', rectified_middle)
# Save the stereo calibration parameters
np.savez('stereo_calibration_dataLM.npz', mtx_left=mtx_left, dist_left=dist_left, 
         mtx_right=mtx_right, dist_right=dist_right, R=R_left_middle, T=T_left_middle, Q=Q)