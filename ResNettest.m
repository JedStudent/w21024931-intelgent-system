load('C:\Users\Jed Bywater\OneDrive - Northumbria University - Production Azure AD\Documents\MATLAB\ResNet18trainedModel.mat', 'net');

% Define the base path
base_path = 'C:\Users\Jed Bywater\OneDrive - Northumbria University - Production Azure AD\Documents\MATLAB\';

% Read test CSV file
calc_case_test_csv = readtable('C:\Users\Jed Bywater\OneDrive - Northumbria University - Production Azure AD\Documents\MATLAB\csv\calc_case_description_test_set.csv');
dicom_metadata = readtable('C:\Users\Jed Bywater\OneDrive - Northumbria University - Production Azure AD\Documents\MATLAB\csv\dicom_info.csv');


% Grab relevant columns
testImagePaths = calc_case_test_csv{:, 'ROI_mask_file_path'};
testPathologyLabels = calc_case_test_csv{:, 'pathology'};
allDicomImagePaths = dicom_metadata{:, 'image_path'};


% Extract subfolder names using regex
numTestPaths = numel(testImagePaths);
testDesiredSubstrings = cell(numTestPaths, 1);

for i = 1:numTestPaths
    path = testImagePaths{i};
    pattern = '(?<=/)[^/]*(?=/)';
    tokens = regexp(path, pattern, 'match');
    if numel(tokens) >= 2
        testDesiredSubstrings{i} = tokens{2};
    else
        disp(['Could not extract the desired substring from path: ', path]);
    end
end

% Match image paths and their corresponding labels for the test data
numTestSubfolders = numel(testDesiredSubstrings);
testMatchedImagePaths = cell(numTestSubfolders, 1);
testMatchedLabels = zeros(numTestSubfolders, 1);

for i = 1:numTestSubfolders
    subfolder = testDesiredSubstrings{i};
    matchedIndex = contains(allDicomImagePaths, subfolder);
    
    if sum(matchedIndex) >= 1
        testMatchedImagePaths{i} = [base_path, allDicomImagePaths{find(matchedIndex, 1)}]; % Concatenate base path with the matched image path
        testMatchedLabels(i) = strcmp(testPathologyLabels{i}, 'MALIGNANT');
    else
        disp(['No matching image path found for subfolder: ', subfolder]);
    end
end

% Print matched image paths and their corresponding labels for testing
for i = 1:numTestSubfolders
    fprintf('Matched Image Path: %s\n', testMatchedImagePaths{i});
    fprintf('Label: %d\n', testMatchedLabels(i));
    fprintf('-------------------------\n');
end

% Remove non-existing files from the dataset
testExists = cellfun(@(x) isfile(x), testMatchedImagePaths);
testFilteredImagePaths = testMatchedImagePaths(testExists);
testFilteredLabels = testMatchedLabels(testExists);

% Check if all files exist
testInvalidFiles = cellfun(@(x) ~isfile(x), testFilteredImagePaths);
if any(testInvalidFiles)
    fprintf('The following files do not exist:\n');
    for i = 1:numel(testInvalidFiles)
        if testInvalidFiles(i)
            fprintf('%s\n', testFilteredImagePaths{i});
        end
    end
    error('One or more files do not exist. Please check the file paths.');
end 

% Load the ResNet18 model
load('C:\Users\Jed Bywater\OneDrive - Northumbria University - Production Azure AD\Documents\MATLAB\ResNet18trainedModel.mat', 'net');

% Assuming the same CSV files and image paths

% Create an imageDatastore and resize the images for the test data
imageSize = [224, 224, 3];
testImds = imageDatastore(testFilteredImagePaths, 'Labels', categorical(testFilteredLabels), 'ReadFcn', @(x) imresize(repelem(imread(x), 1, 1, 3), imageSize(1:2)));

% Find the index of the last fully connected layer just before the softmaxLayer
lastFCLayerIndex = find(arrayfun(@(x) isa(x, 'nnet.cnn.layer.FullyConnectedLayer'), net.Layers), 1, 'last');
lastFCLayerName = net.Layers(lastFCLayerIndex).Name;

% Get the class output scores for the last fully connected layer
disp(numel(testImds.Files));
testYScores = activations(net, testImds, lastFCLayerName);

% Calculate the predicted labels
testYScoresReshaped = reshape(testYScores, [], size(testYScores, 4))';
[~, maxIdx] = max(testYScoresReshaped, [], 2);
classNames = {'0', '1'}; % Assuming '0' for benign and '1' for malignant
testYPred = categorical(cellstr(classNames(maxIdx)));

% Get testYValidation from testImds
testYValidation = testImds.Labels;

% Calculate the test accuracy
testAccuracy = sum(testYPred == testYValidation) / numel(testYValidation);
fprintf('Test accuracy: %.2f%%\n', testAccuracy * 100);

% Obtain the probabilities for each class
testYScores = predict(net, testImds);

% Change categorical predictions to numeric for binary format
[~,maxIdx] = max(testYScores,[],2);
testYPred = categorical(maxIdx-1); % '1' for malignant and '0' for benign assuming

% Calculate softmax probabilities
testYProbabilities = exp(testYScores) ./ sum(exp(testYScores), 2);

% Calculate the ROC curve and AUC
testYValidationBin = double(testYValidation == '1'); % Convert the test labels to binary format
[testFPR, testTPR, ~, testAUC] = perfcurve(testYValidationBin, testYProbabilities(:, 2), 1);

% Plot the ROC curve
figure;
plot(testFPR, testTPR);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ROC Curve (AUC = %.2f)', testAUC));

% Plot the confusion matrix for the test dataset
figure
confusionchart(testYValidation, testYPred);
title('Confusion Matrix for Test Data');
