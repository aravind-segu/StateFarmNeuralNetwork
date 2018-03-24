pkg load image;
more off;
trainDataPath = 'C:/Users/aravi/Documents/Aravind/StateFarmNeuralNetwork/StateFarm/Images/train';

% Process Images
printf('Processing Images...\n')

images = imageProcessing(trainDataPath);
