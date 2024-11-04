function [] = mesh_create_func(img, disparityMap, xyzPoints,unreliable)
%Matlab code for creating a 3D surface mesh
%% create a connectivity structure
[M, N] = size(disparityMap); % get image size
res = 2; % resolution of mesh
[nI,mI] = meshgrid(1:res:N,1:res:M); % create a 2D meshgrid of pixels, thus defining a resolution grid
TRI = delaunay(nI(:),mI(:)); % create a triangle connectivity list
indI = sub2ind([M,N],mI(:),nI(:)); % cast grid points to linear indices
%% linearize the arrays and adapt to chosen resolution
pcl = reshape(xyzPoints,N*M,3); % reshape to (N*M)x3
J1l = reshape(img,N*M,3); % reshape to (N*M)x3
pcl = pcl(indI,:); % select 3D points that are on resolution grid
J1l = J1l(indI,:); % select pixels that are on the resolution grid
%% remove the unreliable points and the associated triangles
ind_unreliable = find(unreliable(indI));% get the linear indices of unreliable 3D points
imem = ismember(TRI(:),ind_unreliable); % find indices of references to unreliable points
[ir,~] = ind2sub(size(TRI),find(imem)); % get the indices of rows with refs to unreliable points.
TRI(ir,:) = []; % dispose them
iused = unique(TRI(:)); % find the ind's of vertices that are in use
used = zeros(length(pcl),1); % pre-allocate
used(iused) = 1; % create a map of used vertices
map2used = cumsum(used); % conversion table from indices of old vertices to the new one
pcl = pcl(iused,:); % remove the unused vertices
J1l = J1l(iused,:);
TRI = map2used(TRI); % update the ind's of vertices
%% Filter out rows with Inf or NaN values
validRows = all(isfinite(pcl), 2);   % Check for rows with only finite values
pcl = pcl(validRows, :);             % Keep only valid rows
J1l = J1l(validRows, :);             % Keep corresponding color data

% Debug output: Check number of points and color data alignment
disp(['Number of 3D points after filtering: ', num2str(size(pcl, 1))]);
disp(['Number of color data points after filtering: ', num2str(size(J1l, 1))]);

% Create a mapping from old indices to new indices
map2used = zeros(length(indI), 1);
map2used(validRows) = 1:sum(validRows);  % Map valid points to new indices

%% Update TRI to only include valid triangles
TRI = TRI(all(ismember(TRI, find(validRows)), 2), :); % Keep valid triangles
TRI = map2used(TRI);  % Update the indices in TRI to refer to the new mapping

% Debug output: Check TRI for invalid indices
validPoints = unique(TRI(:));
if max(validPoints) > size(pcl, 1)
    error('TRI references indices that exceed the number of valid points');
end
%% create the 3D mesh
TR = triangulation(TRI,double(pcl)); % create the object
%% visualize
figure;
TM = trimesh(TR); % plot the mesh
set(TM,'FaceVertexCData',J1l); % set colors to input image
set(TM,'Facecolor','interp');
% set(TM,'FaceColor','red'); % if you want a colored surface
set(TM,'EdgeColor','none'); % suppress the edges
xlabel('x (mm)')
ylabel('y (mm)')
zlabel('z (mm)')
axis([-250 250 -250 250 400 900])
set(gca,'xdir','reverse')
set(gca,'zdir','reverse')
daspect([1,1,1])
axis tight
