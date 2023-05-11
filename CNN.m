% Define the base path
base_path = 'C:\Users\Jed Bywater\OneDrive - Northumbria University - Production Azure AD\Documents\MATLAB\';

% Read CSV files
calc_case_train_csv = readtable('C:\Users\Jed Bywater\OneDrive - Northumbria University - Production Azure AD\Documents\MATLAB\csv\calc_case_description_train_set.csv');
dicom_metadata = readtable('C:\Users\Jed Bywater\OneDrive - Northumbria University - Production Azure AD\Documents\MATLAB\csv\dicom_info.csv');

% Grab relevant columns
imagePaths = calc_case_train_csv{:, 'ROI_mask_file_path'};
pathologyLabels = calc_case_train_csv{:, 'pathology'};
allDicomImagePaths = dicom_metadata{:, 'image_path'};

% Extract subfolder names using regex
numPaths = numel(imagePaths);
desiredSubstrings = cell(numPaths, 1);

for i = 1:numPaths
    path = imagePaths{i};
    pattern = '(?<=/)[^/]*(?=/)';
    tokens = regexp(path, pattern, 'match');
    if numel(tokens) >= 2
        desiredSubstrings{i} = tokens{2};
    else
        disp(['Could not extract the desired substring from path: ', path]);
    end
end

% Match image paths and their corresponding labels
numSubfolders = numel(desiredSubstrings);
matchedImagePaths = cell(numSubfolders, 1);
matchedLabels = zeros(numSubfolders, 1);

for i = 1:numSubfolders
    subfolder = desiredSubstrings{i};
    matchedIndex = contains(allDicomImagePaths, subfolder);
    
    if sum(matchedIndex) >= 1
        matchedImagePaths{i} = [base_path, allDicomImagePaths{find(matchedIndex, 1)}]; % Concatenate base path with the first matched image path
        matchedLabels(i) = strcmp(pathologyLabels{i}, 'MALIGNANT');
    else
        disp(['No matching image path found for subfolder: ', subfolder]);
    end
end

% Print matched image paths and their corresponding labels for testing
for i = 1:numSubfolders
    fprintf('Matched Image Path: %s\n', matchedImagePaths{i});
    fprintf('Label: %d\n', matchedLabels(i));
    fprintf('-------------------------\n');
end

% Remove non-existing files from the dataset
exists = cellfun(@(x) isfile(x), filteredImagePaths);
filteredImagePaths = filteredImagePaths(exists);
filteredLabels = filteredLabels(exists);
% Check if all files exist
invalidFiles = cellfun(@(x) ~isfile(x), filteredImagePaths);
if any(invalidFiles)
    fprintf('The following files do not exist:\n');
    for i = 1:numel(invalidFiles)
        if invalidFiles(i)
            fprintf('%s\n', filteredImagePaths{i});
        end
    end
    error('One or more files do not exist. Please check the file paths.');
end


% Create an imageDatastore and resize the images
imageSize = [224, 224, 3];
imds = imageDatastore(filteredImagePaths, 'Labels', categorical(filteredLabels), ...
    'ReadFcn', @(x) imresize(repelem(imread(x), 1, 1, 3), imageSize(1:2)));
% Split the data into training and validation sets
[trainImgs, valImgs] = splitEachLabel(imds, 0.8, 'randomized');

% Create a simple CNN architecture
layers = [
    imageInputLayer(imageSize)
    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

% Specify training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 20, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', valImgs, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');


% Train the CNN model
net = trainNetwork(trainImgs, layers, options);

% Evaluate the trained model on the validation set
YPred = classify(net, valImgs);
YValidation = valImgs.Labels;
accuracy = sum(YPred == YValidation) / numel(YValidation);
fprintf('Validation accuracy: %.2f%%\n', accuracy * 100);

% Plot the confusion matrix
figure
confusionchart(YValidation, YPred);
title('Confusion Matrix for Validation Data');


save('C:\Users\Jed Bywater\OneDrive - Northumbria University - Production Azure AD\Documents\MATLAB\CNNtrainedModel.mat', 'net');


