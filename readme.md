## Facial Keypoints Prediction in Deep Learning (Fall 2016)<br />

### --- Objective ---<br />
The objective of this task is using Convolutional Neural Network(CNN) to predict keypoint positions on face images.

### --- Data description ---<br />
The data is from Kaggle, which contains a set of face images. The input image consists of a list of 96*96 pixels. 
https://www.kaggle.com/c/facial-keypoints-detection

Each predicted keypoint is specified by an (x,y) real-valued pair in the space of pixel indices. 15 keypoints in the dataset represent the different elements of the face (left_eye_center, right_eye_center, nose_tip …);

The input image is given in the last field of the data files, consists of a list of pixels, as integers in (0,255). The images are 96x96 pixels;

The training data contains a list of 7049 images. Each row contains the (x,y) coordinates for 15 keypoints, and image data as row-ordered list of pixels:<br />  
![Alt text]( training_data.png?raw=true "")<br />  

Test data contains a list of 1783 test images. Each row contains ImageId and image data as row-ordered list of pixels:<br />  
![Alt text]( test_data.png?raw=true "")<br />  

### --- CNN Model ---<br />
The formula for calculating the output size for any given convolutional layer is<br />
![Alt text]( cnn_layer_compute.jpg?raw=true "")<br />
where O is the output height/length, W is the input height/length, K is the filter size, P is the padding, and S is the stride.<br />
![Alt text]( cnn_model.jpg?raw=true "")<br />

The detailed design in CNN model:<br />
The input image in the training set is 96*96 pixels;<br />

The first convolutional layer contains 32 filters with a 3*3 kernel, the stride is 1. The output feature map is 94*94. A a rectified linear unit (ReLU) is followed after the convolutional layer. And then, a maxpool layer followed the ReLU layer. And a 2*2 max pool is used with stride 2. so the output feature map is 47*47.<br />

The second convolutional layer contains 64 filters with a 2*2 kernel, the stride is 1. The output feature map is 46*46. Then a ReLU is followed, and following with a maxpool layer with a 2*2 max pool and a stride of 2. so the output feature map is 23*23.<br />

The second convolutional layer contains 64 filters with a 2*2 kernel, the stride is 1. The output feature map is 22*22. Then a ReLU is followed, and following with a maxpool layer with a 2*2 max pool and a stride of 2. so the output feature map is 11*11.<br />

After the 3 convolutional layers, 3 fully connected layers followed. Each layer has 500, 500 and 30 outputs.<br />

### --- Optimizer ---<br />
Stochastic Gradient Descent (SGD) and its variants are probably the most used optimization algorithms for machine learning in general and for deep learning in particular.<br />
![Alt text]( sgd.jpg?raw=true "")<br />
Adam is a method for efficient stochastic optimization that only requires first-order gradients with little memory requirement. Adam Optimizer is used in this project.

### --- Evaluation ---<br />
Submissions are scored on the Root Mean Squared Error(RMSE)<br />
![Alt text]( rmse.jpg?raw=true "")<br />

### --- Result visualization ---<br />
![Alt text]( result.png?raw=true "")<br />
