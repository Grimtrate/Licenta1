%%Incarcarea imaginilor de antrenare
allImages = imageDatastore('D:\stuff machine learning\jpg', 'IncludeSubfolders', true,...
   'LabelSource', 'foldernames');
tbl = countEachLabel(allImages);
%% Impartirea imaginilor in imagini de antrenare si imagini de test
[trainingImages, testImages] = splitEachLabel(allImages, 0.9, 'randomize');
%% Definirea layerelor pentru CNN
conv1 = convolution2dLayer(11,96,'Stride',4,'Padding',0); %290.5k neuroni
conv2 = convolution2dLayer(5,256,'Stride',1,'Padding',2); %7milioane neuroni
conv3 = convolution2dLayer(3,384,'Stride',1,'Padding',1);
conv4 = convolution2dLayer(3,384,'Stride',1,'Padding',1);
conv5 = convolution2dLayer(3,256,'Stride',1,'Padding',1);
layers = [...
    imageInputLayer([227 227 3]);
    conv1;
    batchNormalizationLayer('Name','batch1');
    reluLayer('Name','relu1');
    maxPooling2dLayer(3,'Name','pool1','Stride',2);
    conv2;
     batchNormalizationLayer('Name','batch2');
    reluLayer('Name','relu2');
    maxPooling2dLayer(3,'Name','pool2','Stride',2);
    conv3;
    batchNormalizationLayer('Name','batch3');
    reluLayer('Name','relu3');
    conv4;
    batchNormalizationLayer('Name','batch4');
    reluLayer('Name','relu4');
    conv5;
    batchNormalizationLayer('Name','batch5');
    reluLayer('Name','relu5');
    maxPooling2dLayer(3,'Name','pool5','Stride',2);
    fullyConnectedLayer(4096,'Name','fc6');
    reluLayer('Name','relu6');
    dropoutLayer('Name','drop6');
    fullyConnectedLayer(4096,'Name','fc7');
    reluLayer('Name','relu8');
    dropoutLayer('Name','drop7');
    fullyConnectedLayer(103,'Name','fc8');
    softmaxLayer('Name','prob');
    classificationLayer('Name','output');]
%% Setarea optiunilor pentru training
opts = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 10, ...
    'L2Regularization', 1.0000e-04, ...
    'MaxEpochs', 80, ...
    'MiniBatchSize', 256, ...
    'ValidationData',testImages, ...
    'CheckpointPath','D:\stuff machine learning\checkpoints', ...
    'Verbose', true,...
    'Plot','training-progress');

%% Modificarea pozelor la dimensiunile asteptate folosind o functie custom
testImages.ReadFcn = @readFunctionTrain1;
trainingImages.ReadFcn = @readFunctionTrain1;
%% Antrenarea retelei
myNet = trainNetwork(trainingImages, layers, opts);
save('trainedNet.mat','myNet')
save('testImages.mat','testImages')
%% Classify Validation Images
[YPred,probs] = classify(myNet,testImages);
accuracy = mean(YPred == testImages.Labels)
%% Afiseaza 4 exemple cu predictie
idx = randperm(numel(testImages.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(testImages,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end
%% Testare live
camera = webcam; % Connect to the camera
net = myNet; % Load the neural net

while true   
    picture = camera.snapshot;              % Take a picture    
    picture = imresize(picture,[227,227]);  % Resize the picture

   label = classify(net, picture);        % Classify the picture
       
    image(picture);     % Show the picture
    title(char(label)); % Show the label
    drawnow;   
end
%%
plotconfusion(trainingImages.Labels,YPred)