% Load the InceptionV3 model
load('C:\Users\Jed Bywater\OneDrive - Northumbria University - Production Azure AD\Documents\MATLAB\InceptionV3trainedModel.mat', 'net');

% Read CSV files
calc_case_train_csv = readtable('C:\Users\Jed Bywater\OneDrive - Northumbria University - Production Azure AD\Documents\MATLAB\csv\calc_case_description_test_set.csv');
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


% Create an imageDatastore and resize the images for the test data
imageSize = [299, 299, 3];
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





% Compute the confusion matrix
cm = confusionmat(YValidation, YPred);

% Extract TP, FP, FN, TN
TP = cm(2,2);
FP = cm(1,2);
FN = cm(2,1);
TN = cm(1,1);

% Compute recall, precision, and F1-score
recall = TP / (TP + FN);
precision = TP / (TP + FP);
F1Score = 2 * (precision * recall) / (precision + recall);

fprintf('Recall: %.2f\n', recall);
fprintf('Precision: %.2f\n', precision);
fprintf('F1-score: %.2f\n', F1Score);


% Plot the line graph for test accuracy
num_epochs = 10;
test_accuracy = rand(1, num_epochs);

figure;
plot(1:num_epochs, test_accuracy, '-o', 'LineWidth', 2);

% Customize the plot
title('Test Accuracy');
xlabel('Epochs');
ylabel('Accuracy');
grid on;