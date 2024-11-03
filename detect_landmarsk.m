% Step 1: Load the image
I_left = imread('SubjectPictures\subject1\subject1Left\subject1_Left_1.jpg');  % Replace with your image file
I_middle = imread('SubjectPictures\subject1\subject1Middle\subject1_Middle_1.jpg');  % Replace with your image file
I_right = imread('SubjectPictures\subject1\subject1Right\subject1_Right_1.jpg');  % Replace with your image file

function [points] = detectLandmarks(I)
% Step 2: Create a face detector object
faceDetector = vision.CascadeObjectDetector();

% Detect the face in the image
bbox = step(faceDetector, I);

% Step 3: Detect facial landmarks (eyes, nose, mouth, etc.)
% Create a point tracker for facial landmark detection
landmarkDetector = vision.PointTracker('MaxBidirectionalError', 2);

% Initialize the point tracker with detected points (customize for more points)
points = detectMinEigenFeatures(rgb2gray(I), 'ROI', bbox(1,:));  % Initial feature points in the bounding box
points = points.Location;

% Initialize the point tracker
initialize(landmarkDetector, points, I);

% Display the points on the image
plot(points(:,1), points(:,2), 'go', 'MarkerSize', 5);
title('Facial Landmark Points Detected');
hold off;
end


left_features = detectLandmarks(I_left);
% Initialize the point tracker
initialize(landmarkDetector, left_features, I_left);
% Display the points on the image
plot(points(:,1), points(:,2), 'go', 'MarkerSize', 5);
title('Facial Landmark Points Detected');
hold off;

mid_features = detectLandmarks(I_middle);
figure
right_features = detectLandmarks(I_right);