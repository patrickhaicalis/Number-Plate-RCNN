%clear;
%clc; 

load('numberplatetable')
load('stopSign')
load('layers')
load('trainingsets')


imDir = fullfile('C:\Users\patri\Documents\MATLAB\FYP\Number Plates');
addpath(imDir);

options = trainingOptions('sgdm', ...
  'MiniBatchSize', 32, ... %32
  'InitialLearnRate', 1e-6, ...
  'MaxEpochs', 5);  %10

rcnn = trainFasterRCNNObjectDetector(stopSign, layers, options, 'NegativeOverlapRange', [0 0.5]);

img = imread('plate071.JPG'); 

[bbox, score, label] = detect(rcnn, img, 'MiniBatchSize', 32);
[score, idx] = max(score);

bbox = bbox(idx, :);
annotation = sprintf('%s: (Confidence = %f)', label(idx), score);

detectedImg = insertObjectAnnotation(img, 'rectangle', bbox, annotation);

figure
imshow(detectedImg)

rmpath(imDir); 