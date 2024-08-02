clc
clear all

% Upload the image
[filename, filepath] = uigetfile('*.png', 'Select the required image file');
if isequal(filename, 0)
    disp('No file selected');
else
    mainFunction(fullfile(filepath, filename));
end

function mainFunction(filename)
    % Load and preprocess image
    img = loadImage(filename);
    figure();imshow(img)
    
    % Find circle locations
    circles = findCircles(img);

    % Check if at least 4 circles are detected
    if size(circles, 1) < 4
        disp('Less than 4 circles are detected. Therefore, cannot perform geometric transformation.');
        return;
    end

    % Correct image distortion based on circles
    [adjusted_img, circles] = correctImage(circles, img);

    % Display the corrected image
    imshow(adjusted_img);
    title('Corrected Image');

    % Identify colors within specific regions
    color_matrix = findColors(adjusted_img);
    disp("COLOUR-MATRIX OF IMAGE:");
    disp(color_matrix); % Display color matrix
end

%% Find the colors in the image
function color_matrix = findColors(img)
    lab_img = rgb2lab(img); % Converting RGB colorspace to LAB
    f = fspecial('average', 11); % Using mean-filter (f) to remove noise
    lab_img = imfilter(lab_img, f);

    % Coordinates (c) from where we find the colors
    c = [80 172 259 369];

    % Finding all 16 points
    color_points = zeros(16, 3);
    count = 0;
    for i = 1:4
        for j = 1:4
            count = count + 1;
            x = c(1, i);
            y = c(1, j);
            temp = lab_img(x:x + 56, y:y + 56, :);
            color_points(count, : ) = mean(reshape(temp, [], 3), 1);
        end
    end

    % Defining the colors in RGB and LAB
    rgb_scale = [1 0 0; 0 1 0; 0 0 1; 1 1 0; 1 1 1; 0.2824, 0.2392, 0.5451]; % Red, Green, Blue, Yellow, White, Violet
    color_names = {'r', 'g', 'b', 'y', 'w', 'p'}; % r=red, g=green, b=blue, y=yellow, w=white, p=purple
    lab_scale = rgb2lab(rgb_scale); % converting the colors to lab

    % Calculating distances between color points and color scale
    d = pdist2(color_points, lab_scale, 'euclidean');

    % Assigning colors based on minimum distances
    [~, idx] = min(d, [], 2);
    patchnames = color_names(idx);

    % Reshaping the color matrix
    color_matrix = reshape(patchnames, 4, 4)';
end


% Loads an image from the file specified by filename, and returns it as type double.
function image = loadImage(filename)
    % Read in the image using imread
    img = imread(filename);
    % Convert the image to double precision
    image = im2double(img);
    figure();imshow(image)
    title('Input Image');
end

% Find the coordinates of the black circle
function circleCoordinates = findCircles(image)
    img = image;    
    gray_img = rgb2gray(img); % Changing the given image to grey image
    threshold = graythresh(gray_img);
    binary_img = imbinarize(gray_img, threshold);
    inverted_binary_img = imcomplement(binary_img); % inverting the binary image
    cc = bwconncomp(inverted_binary_img);
    areas = cellfun(@numel, cc.PixelIdxList);
    [sorted_areas, sorted_indices] = sort(areas, 'descend'); % Sorting in descending order

    % Getting the coordinates of the first four largest black blobs
    num_blobs = 5;
    blob_coords = zeros(num_blobs, 2);
    for i = 2:num_blobs
        blob_indices = cc.PixelIdxList{sorted_indices(i)};
        [rows, cols] = ind2sub(size(inverted_binary_img), blob_indices);
        blob_coords(i, :) = [mean(cols), mean(rows)];
    end

    % Removing the first coordinate from the blob_coords matrix
    blob_coords(1, :) = [];
    % Sort the coordinates in clockwise order starting from bottom-left
    sortedCoordinates = sortrows(blob_coords);

    if sortedCoordinates(2, 2) < sortedCoordinates(1, 2)
        % If the second coordinate is below the first, swap them
        sortedCoordinates([1 2], :) = sortedCoordinates([2 1], :);
    end
    if sortedCoordinates(4, 2) > sortedCoordinates(3, 2)
        % If the fourth coordinate is above the third, swap them
        sortedCoordinates([3 4], :) = sortedCoordinates([4 3], :);
    end
    circleCoordinates = sortedCoordinates;
end

% Correct the distorted images
function [outputImage, correctedCoordinates] = correctImage(coordinates, image)
    boxf = [[0, 0]; [0, 480]; [480, 480]; [480, 0]]; % Define a fixed box with coordinates
    disp(coordinates)

    % Calculating the transformation matrix from the given Coordinates to transform the matrix to the fixed box using projective transformation
    TF = fitgeotrans(coordinates, boxf, 'projective');
    % Create an image reference object with the size of the input image
    outview = imref2d(size(image));
    % Apply the calculated transformation matrix to the input image and create a new image with fill value 255 (white) outside the boundaries of the input image
    B = imwarp(image, TF, 'fillvalues', 255, 'OutputView', outview);
    % Crop the image to a size of 480x480
    B = imcrop(B, [0, 0, 480, 480]);
    % Try to suppress the glare in the image using flat-field correction
    B = imflatfield(B, 40);
    % Adjust the levels of the image to improve contrast
    B = imadjust(B, [0.4, 0.65]);
    % Assign the corrected image to the outputImage variable
    outputImage = B;
    correctedCoordinates = boxf;
end