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
% Check if all files exist
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


% Your existing code (up to the point where you create the imageDatastore)

% Create an imageDatastore and resize the images
imageSize = [299, 299, 3];
imds = imageDatastore(filteredImagePaths, 'Labels', categorical(filteredLabels), ...
    'ReadFcn', @(x) imresize(convertTo3Channels(imread(x)), imageSize(1:2)));

% Split the data into training and validation sets
[trainImgs, valImgs] = splitEachLabel(imds, 0.8, 'randomized');

% Load the pretrained Inception-v3
net = inceptionv3;

% Replace the last two layers to match the number of classes
lgraph = layerGraph(net);
numClasses = 2;
newLayers = [fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
             softmaxLayer('Name','softmax')
             classificationLayer('Name','classoutput')];
lgraph = replaceLayer(lgraph,'predictions',newLayers(1));
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newLayers(3));

% Remove the existing input layer
lgraph = removeLayers(lgraph, 'input_1');

% Add new input layer to the beginning of the network
inputSize = [299, 299, 3];
inputLayer = imageInputLayer(inputSize, 'Name', 'input');
lgraph = addLayers(lgraph, inputLayer);
lgraph = connectLayers(lgraph, 'input', 'conv2d_1');

% Specify training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 8, ... % Try reducing this value
    'Shuffle', 'every-epoch', ...
    'ValidationData', valImgs, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');


% Train the Inception-v3 model
net = trainNetwork(trainImgs, lgraph, options);

% Evaluate the trained model on the validation set
YPred = classify(net, valImgs);
YValidation = valImgs.Labels;
accuracy = sum(YPred == YValidation) / numel(YValidation);
fprintf('Validation accuracy: %.2f%%\n', accuracy * 100);

% Plot the confusion matrix
figure
confusionchart(YValidation, YPred);
title('Confusion Matrix for Validation Data');

% Save the trained model
save('C:\Users\Jed Bywater\OneDrive - Northumbria University - Production Azure AD\Documents\MATLAB\InceptionV3trainedModel.mat', 'net');

% Define the convertTo3Channels function at the end of the script
function img3ch = convertTo3Channels(img)
    if size(img, 3) == 1
        img3ch = repmat(img, 1, 1, 3);
    else
        img3ch = img;
    end
end
