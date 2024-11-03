function [] = mesh_create_func(img, disparityMap, xyzPoints)
    %% Create a connectivity structure
    [M, N] = size(disparityMap);        % Get image size
    res = 2;                            % Resolution of mesh   
    [nI, mI] = meshgrid(1:res:N, 1:res:M); % Create a 2D meshgrid of pixels, defining a resolution grid
    TRI = delaunay(nI(:), mI(:));       % Create a triangle connectivity list
    indI = sub2ind([M, N], mI(:), nI(:));  % Cast grid points to linear indices

    %% Linearize the arrays and adapt to chosen resolution
    pcl = reshape(xyzPoints, N * M, 3);  % Reshape to (N*M)x3
    J1l = reshape(img, N * M, 3);        % Reshape to (N*M)x3

    % Convert color data to double if necessary
    if ~isa(J1l, 'double')
        J1l = im2double(J1l); % Ensure range is [0, 1]
    end

    pcl = pcl(indI, :);                  % Select 3D points on the resolution grid
    J1l = J1l(indI, :);                  % Select pixels on the resolution grid

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

    %% Create the 3D mesh
    TR = triangulation(TRI, double(pcl));

    %% Visualize
    figure;
    TM = trisurf(TR);
    set(TM, 'FaceVertexCData', J1l);     % Set colors to input image for a realistic appearance
    set(TM, 'Facecolor', 'interp');      % Use interpolated coloring for smooth appearance
    set(TM, 'EdgeColor', 'none');        % Suppress edges for smooth visualization
    xlabel('x (mm)');
    ylabel('y (mm)');
    zlabel('z (mm)');
    axis([-250 250 -250 250 400 900]);   % Set axis limits
    set(gca, 'xdir', 'reverse');         % Reverse x-axis for correct orientation
    set(gca, 'zdir', 'reverse');         % Reverse z-axis for correct orientation
    daspect([1, 1, 1]);                  % Set data aspect ratio
    axis tight;                          % Fit axis to data
end
