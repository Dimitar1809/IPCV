import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# Define the function to detect foreground
def detect_foreground(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Edge Detection
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)

    # Step 2: Morphological Operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated_edges = cv2.dilate(edges, kernel, iterations=3)
    closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel)
    cleaned_edges = cv2.morphologyEx(closed_edges, cv2.MORPH_OPEN, kernel)

    # Step 3: Create a Mask
    contours, _ = cv2.findContours(cleaned_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Step 4: Apply the Mask
    foreground = cv2.bitwise_and(image, image, mask=mask)
    background = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    
    return foreground, background

# Load the rectified images
rectified_left = cv2.imread('rectified_left.jpg')
rectified_right = cv2.imread('rectified_middle.jpg')


# Detect the foreground and background
foregroundLeft, backgroundLeft = detect_foreground(rectified_left)
foregroundRight, backgroundRight = detect_foreground(rectified_right)
# downscale images for faster processing
image_left = cv2.pyrDown(foregroundLeft)  
image_right = cv2.pyrDown(foregroundRight)

# Turn the foreground to grayscale
foregroundLeft = cv2.cvtColor(foregroundLeft, cv2.COLOR_BGR2GRAY)
foregroundRight = cv2.cvtColor(foregroundRight, cv2.COLOR_BGR2GRAY)

# # Display the results
# cv2.imshow("Foreground", foregroundLeft)
# cv2.imshow("Background", backgroundLeft)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Step 1: Stereo Matching (Disparity Map Calculation)
# For each pixel algorithm will find the best disparity from 0
# Larger block size implies smoother, though less accurate disparity map

# Set disparity parameters
# Note: disparity range is tuned according to specific parameters obtained through trial and error. 
block_size = 9
min_disp = 0
max_disp = 512
num_disp = max_disp - min_disp # Needs to be divisible by 16

# Create Block matching object. 
# stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
# 	numDisparities = num_disp,
# 	blockSize = block_size,
# 	uniquenessRatio = 10,
# 	speckleWindowSize = 10,
# 	speckleRange = 2,
# 	disp12MaxDiff = 1,
# 	P1 = 8 * 3 * block_size**2,#8*img_channels*block_size**2,
# 	P2 = 32 * 3 * block_size**2) #32*img_channels*block_size**2)


stereo = cv2.StereoBM_create(numDisparities=512, blockSize=5)

# Compute the disparity map
disparity_map = stereo.compute(foregroundLeft, foregroundRight).astype(np.float32) / 16.0


# Show disparity map before generating 3D cloud to verify that point cloud will be usable. 
plt.imshow(disparity_map,'gray')
plt.show()




# Step 2: Depth Map Calculation
# Assuming these calibration parameters from the left camera
focal_length= np.load('stereo_calibration_dataLM.npz')['mtx_left'][0, 0]  # fx of the left camera
baseline = np.linalg.norm(np.load('stereo_calibration_dataLM.npz')['T'])  # Baseline in meters
Q = np.load('stereo_calibration_dataLM.npz')['Q']  # Disparity-to-depth mapping matrix

# Get new downsampled width and height 
h,w = rectified_right.shape[:2]

# Convert disparity map to float32 and divide by 16 as show in the documentation
print(disparity_map.dtype)
#disparity_map = np.float32(np.divide(disparity_map, 16.0))
print(disparity_map.dtype)

# Reproject points into 3D
points_3D = cv2.reprojectImageTo3D(disparity_map, Q, handleMissingValues=True)
# Get color of the reprojected points
colors = cv2.cvtColor(foregroundRight, cv2.COLOR_BGR2RGB)

# Get rid of points with value 0 (no depth)
mask_map = disparity_map > disparity_map.min()

# Mask colors and points. 
output_points = points_3D[mask_map]
output_colors = colors[mask_map]


# Function to create point cloud file
def create_point_cloud_file(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')


output_file = 'pointCloud.ply'

# Generate point cloud file
create_point_cloud_file(output_points, output_colors, output_file)





# depth_mapLM = (focal_length * baseline) / (disparity + 1e-6)  # Depth map in meters
# # Mask out areas with very high depth values (likely due to near-zero disparity)
# depth_mapLM[depth_mapLM > 3000] = 0  # Set an arbitrary upper limit for visualization (e.g., 10 meters)

# # Display the depth map with color normalization
# plt.imshow(depth_mapLM, cmap='viridis', vmin=0, vmax=10)  # Display values between 0 and 10 meters
# plt.colorbar()
# plt.title("Depth Map")
# plt.show()
# # Step 3: 3D Point Cloud Generation
# h, w = depth_mapLM.shape
# points = []
# colors = []

# for v in range(h):
#     for u in range(w):
#         depth = depth_mapLM[v, u]
#         if depth > 0 and depth < 10:  # Filter points by depth range (0 < depth < 10 meters)
#             z = depth
#             x = (u - w / 2) * z / focal_length
#             y = (v - h / 2) * z / focal_length
#             points.append([x, y, z])

#             # Use the color from the rectified left image
#             color = rectified_left[v, u] / 255.0  # Normalize color to [0, 1]
#             colors.append([color, color, color])

# # Convert lists to NumPy arrays
# points = np.array(points)
# colors = np.array(colors)

# # Create Open3D point cloud and visualize
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# pcd.colors = o3d.utility.Vector3dVector(colors)
# o3d.visualization.draw_geometries([pcd], window_name="3D Face Reconstruction")