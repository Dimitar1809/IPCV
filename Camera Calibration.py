import cv2
import numpy as np
import glob

CHECKERBOARD = (9, 6)
image_shape = None  

images_left = sorted(glob.glob('CalibrationPictures/Calibratie 1/calibrationLeft/*.jpg'))
images_middle = sorted(glob.glob('CalibrationPictures/Calibratie 1/calibrationMiddle/*.jpg'))
images_right = sorted(glob.glob('CalibrationPictures/Calibratie 1/calibrationRight/*.jpg'))

# Criteria for termination of the iterative algorithm. Either after 30 iterations or if the change in result is smaller than 0.001 (required accuracy epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D points in real world space (the chessboard corners in a 3D space)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

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

        # Display the corners (optional)
        # cv2.drawChessboardCorners(left_img, CHECKERBOARD, corners_left, ret_left)
        # cv2.drawChessboardCorners(middle_img, CHECKERBOARD, corners_middle, ret_middle)
        # cv2.drawChessboardCorners(right_img, CHECKERBOARD, corners_right, ret_right)

        # cv2.imshow('Left Camera Corners', left_img)
        # cv2.imshow('Middle Camera Corners', middle_img)
        # cv2.imshow('Right Camera Corners', right_img)
        # cv2.waitKey(500)

cv2.destroyAllWindows()

# Perform camera calibration for each camera
print("Calibrating Left Camera...")
ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints, imgpoints_left, image_shape, None, None)
print("Calibrating Middle Camera...")
ret_middle, mtx_middle, dist_middle, rvecs_middle, tvecs_middle = cv2.calibrateCamera(objpoints, imgpoints_middle, image_shape, None, None)
print("Calibrating Right Camera...")
ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints, imgpoints_right, image_shape, None, None)

print("\nLeft Camera:")
print(f"Ret: {ret_left}, Matrix: \n{mtx_left}, Distortion Coefficients: \n{dist_left}\n")
print("Middle Camera:")
print(f"Ret: {ret_middle}, Matrix: \n{mtx_middle}, Distortion Coefficients: \n{dist_middle}\n")
print("Right Camera:")
print(f"Ret: {ret_right}, Matrix: \n{mtx_right}, Distortion Coefficients: \n{dist_right}\n")

np.savez('calibration_left.npz', ret=ret_left, mtx=mtx_left, dist=dist_left, rvecs=rvecs_left, tvecs=tvecs_left)
np.savez('calibration_middle.npz', ret=ret_middle, mtx=mtx_middle, dist=dist_middle, rvecs=rvecs_middle, tvecs=tvecs_middle)
np.savez('calibration_right.npz', ret=ret_right, mtx=mtx_right, dist=dist_right, rvecs=rvecs_right, tvecs=tvecs_right)
