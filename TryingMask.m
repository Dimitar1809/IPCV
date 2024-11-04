close all;
% Load pairs of images
img_left = imread('SubjectPictures\subject1\subject1Left\subject1_Left_1.jpg');
img_middle = imread('SubjectPictures\subject1\subject1Middle\subject1_Middle_1.jpg');
img_middle2 = img_middle;
img_right = imread('SubjectPictures\subject1\subject1Right\subject1_Right_1.jpg');

%Convert uint8 to double
img_middle = im2double(img_middle);
img_middle2 = im2double(img_middle2);
img_right = im2double(img_right);
img_left = im2double(img_left);

%% Face Segmentation 
[face_middle, mask_middle] = face_mask_extraction(img_middle,'m');
mask_middle2 = mask_middle;
face_middle2 = face_middle;
[face_right, mask_right] = face_mask_extraction(img_right,'r');
[face_left, mask_left] = face_mask_extraction(img_left,'l');

%% Stereo Rectification 
%--------------------------m-r pair--------------------------------

%Rectify face images
[img_middle_rec,img_right_rec] = rectifyStereoImages(face_middle,face_right,stereoParams_mr, ...
    'OutputView','full');
 figure();imshowpair(img_middle_rec,img_right_rec,'montage');

 %--------------------------m-l pair--------------------------------

%Rectify face images
[img_middle_rec2,img_left_rec] = rectifyStereoImages(face_left,face_middle,stereoParams_ml, ...
    'OutputView','full');
figure();imshowpair(img_middle_rec2,img_left_rec,'montage');

[M2,N2,dummy] = size(img_middle_rec2);  %size of the rectified images

%% Disparity MAP

%Convert RGB image gray-level image
gray_img_m = rgb2gray(img_middle_rec);
gray_img_r = rgb2gray(img_right_rec);
gray_img_m2 = rgb2gray(img_middle_rec2);
gray_img_l = rgb2gray(img_left_rec);

% %%Image smoothing
 h=fspecial('gaussian',5,1);
 gray_img_m = imfilter(gray_img_m,h);
 gray_img_r = imfilter(gray_img_r,h);
 gray_img_m2 = imfilter(gray_img_m2,h);
 gray_img_l = imfilter(gray_img_l,h);


%Disparity PARAMS
disparityRange = [276-6,344+6];
bs = 5;        %defauld bs=15
cTH = 0.7;      %default 0.5
uTH = 15;       %default 15
tTH = 0.0000;   %default 0.0002 only applies if method is blockmatching
dTH = 15;       %default []

%--------------------------m-r pair-------------------------------------
%Block match method
disparityMap1 = disparity(gray_img_m,gray_img_r,'DisparityRange',disparityRange, ...
    'ContrastThreshold',cTH, 'UniquenessThreshold',uTH, 'DistanceThreshold',dTH,'BlockSize',bs, ...
    'Method','BlockMatching');

%Visulizing disparity maps
figure();imshow(disparityMap1,disparityRange);colormap jet;colorbar;

%--------------------------m-l pair-------------------------------------
%Disparity PARAMS
% disparityRange = [276-6,344+6];
% bs = 5;        %defauld bs=15
% cTH = 0.7;      %default 0.5
% uTH = 5;       %default 15
% tTH = 0.0000;   %default 0.0002 only applies if method is blockmatching
% dTH = 5;       %default []

%Block match method
disparityMap2 = disparity(gray_img_m2,gray_img_l,'DisparityRange',disparityRange, ...
    'ContrastThreshold',cTH, 'UniquenessThreshold',uTH, 'DistanceThreshold',dTH,'BlockSize',bs, ...
    'Method','BlockMatching');

%Visulizing disparity maps
figure();imshow(disparityMap2,disparityRange);colormap jet;colorbar;

% %% Unreliable Points
% % unreliable1 = disparityMap1 < -1e+12;
% % %mask_middle_rec = 1-(rgb2gray(img_middle_rec)>0);
% % unreliable1 = unreliable1 | (1-~mask_middle);
% % %figure();imshow(unreliable1,[]);

%% median filtering
disparityMap1 =  medfilt2(disparityMap1,[50 50],"symmetric");
disparityMap2 =  medfilt2(disparityMap2,[50 50],"symmetric");

%% Fill holes
disparityMap1 =  imfill(disparityMap1,"holes");
disparityMap2 =  imfill(disparityMap2,"holes");

%Visulizing disparity maps
figure();imshow(disparityMap1,disparityRange);colormap jet;colorbar;title('Post Processing Disparity MR');
figure();imshow(disparityMap2,disparityRange);colormap jet;colorbar;title('Post Processing Disparity ML');
%% Smoothing disparity maps by Gaussian filter
h=fspecial('gaussian', [20 20], 3);
disparityMap1 = imfilter(disparityMap1,h);
disparityMap2 = imfilter(disparityMap2,h);

%% Find unreliable points
unreliable1 = ones(size(disparityMap1));
unreliable1(find(disparityMap1~=0)) = 0;
unreliable(find(disparityMap1==-realmax('single'))) = 1;

unreliable2 = ones(size(disparityMap2));
unreliable2(find(disparityMap2~=0)) = 0;
unreliable2(find(disparityMap2==-realmax('single'))) = 1;

%% Generate Point Clouds
xyzPoints1 = reconstructScene(disparityMap1,stereoParams_mr);
xyzPoints11 = reconstructScene(disparityMap1,stereoParams_mr);

xyzPoints2 = reconstructScene(disparityMap2,stereoParams_ml);
xyzPoints22 = reconstructScene(disparityMap2,stereoParams_ml);
%% Generate 3D face meshes 
mesh_create_func(img_middle_rec, disparityMap1, xyzPoints1,unreliable1);
mesh_create_func(img_middle_rec2, disparityMap2, xyzPoints2,unreliable2);
%% Merging 2 point clouds and estimate error (ICP algoritm is used)
%Create point cloud object
ptCloud1 = pointCloud(xyzPoints11);
ptCloud2 = pointCloud(xyzPoints22);

%Subsample point clouds by scale
scale= 1;
xyzPoints1_down = pcdownsample(ptCloud1,'random',scale);
xyzPoints2_down = pcdownsample(ptCloud2,'random',scale);

%Obtain rotation matrix from stereoParams
R1 = stereoParams_mr.RotationOfCamera2;
R2 = stereoParams_ml.RotationOfCamera2;

%Define Initilial Transform
tformI =  affine3d();
tformI.T(1:3,1:3) = R2*inv(R1); %R1 is 3x3 matrix so using inv() is okay!

%Transform the point cloud 1 such a way that overlay point cloud 2
[tform,movingReg,rmse] = pcregrigid(xyzPoints1_down,xyzPoints2_down,'MaxIterations',10, ...
   'InitialTransform', tformI);

%Visiluze point clouds
% figure();pcshow(xyzPoints1_down);
% figure();pcshow(xyzPoints2_down);

ptCloudAligned = pctransform(xyzPoints1_down,tform);

%Merge point clouds
ptCloudOut = pcmerge(movingReg,xyzPoints2_down,1);
figure();pcshow(ptCloudOut);