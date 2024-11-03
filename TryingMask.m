% Load pairs of images


img_left = imread('SubjectPictures\subject1\subject1Left\subject1_Left_1.jpg');
img_middle = imread('SubjectPictures\subject1\subject1Middle\subject1_Middle_1.jpg');
img_middle2 = imread('SubjectPictures\subject1\subject1Middle\subject1_Middle_1.jpg');
img_right = imread('SubjectPictures\subject1\subject1Right\subject1_Right_1.jpg');


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
[img_middle_rec2,img_left_rec] = rectifyStereoImages(face_middle2,face_left,stereoParams_ml, ...
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
bs = 15;        %defauld bs=15
cTH = 0.7;      %default 0.5
uTH = 15;       %default 15
tTH = 0.0000;   %default 0.0002 only applies if method is blockmatching
dTH = 15;       %default []

%--------------------------m-r pair-------------------------------------
%SGBM method
disparityMap1 = disparity(gray_img_m,gray_img_r,'DisparityRange',disparityRange, ...
    'ContrastThreshold',cTH, 'UniquenessThreshold',uTH, 'DistanceThreshold',dTH,'BlockSize',bs);

%Block match method
disparityMapBM1 = disparity(gray_img_m,gray_img_r,'DisparityRange',disparityRange, ...
    'ContrastThreshold',cTH, 'UniquenessThreshold',uTH, 'DistanceThreshold',dTH,'BlockSize',bs, ...
    'Method','BlockMatching');

%Visulizing disparity maps
figure();imshow(disparityMapBM1,disparityRange);colormap jet;colorbar;

%--------------------------m-l pair-------------------------------------
%SGBM method
disparityMap2 = disparity(gray_img_m2,gray_img_l,'DisparityRange',disparityRange, ...
    'ContrastThreshold',cTH, 'UniquenessThreshold',uTH, 'DistanceThreshold',dTH,'BlockSize',bs);

%Block match method
disparityMapBM2 = disparity(gray_img_m2,gray_img_l,'DisparityRange',disparityRange, ...
    'ContrastThreshold',cTH, 'UniquenessThreshold',uTH, 'DistanceThreshold',dTH,'BlockSize',bs, ...
    'Method','BlockMatching');

%Visulizing disparity maps
%figure();imshow(disparityMap2,disparityRange);colormap jet;colorbar;
%figure();imshow(disparityMapBM2,disparityRange);colormap jet;colorbar;

%% Unreliable Points
% unreliable1 = disparityMap1 < -1e+12;
% %mask_middle_rec = 1-(rgb2gray(img_middle_rec)>0);
% unreliable1 = unreliable1 | (1-~mask_middle);
% %figure();imshow(unreliable1,[]);

%% median filtering
dsp_med_filt1 =  medfilt2(disparityMapBM1,[50 50]);

%% Smoothing disparity maps by Gaussian filter
h=fspecial('gaussian', [10 10], 2);
dsp_gauss1 = imfilter(dsp_med_filt1,h);

%% Generate Point Clouds
xyzPoints1 = reconstructScene(dsp_gauss1,stereoParams_mr);
xyzPoints11 = reconstructScene(dsp_med_filt1,stereoParams_mr);

%% Generate 3D face meshes 
mesh_create_func( img_middle_rec, dsp_gauss1, xyzPoints1);

%Create point cloud object
ptCloud1 = pointCloud(xyzPoints11);

%Subsample point clouds by scale
scale= 1;
xyzPoints1_down = pcdownsample(ptCloud1,'random',scale);

%Visiluze point clouds
figure();pcshow(xyzPoints1_down);