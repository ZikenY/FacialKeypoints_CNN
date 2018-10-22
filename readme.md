## Facial Keypoints Prediction in Deep Learning (Fall 2016)<br />

### --- Objective ---<br />
The objective of this task is using Convolutional Neural Network(CNN) to predict keypoint positions on face images.

### --- Data description ---<br />
The data is from Kaggle, which contains a set of face images. The input image consists of a list of 96x96 pixels. 
https://www.kaggle.com/c/facial-keypoints-detection

Each predicted keypoint is specified by an (x,y) real-valued pair in the space of pixel indices. 15 keypoints in the dataset represent the different elements of the face (left_eye_center, right_eye_center, nose_tip …);

The input image is given in the last field of the data files, consists of a list of pixels, as integers in (0,255). The images are 96x96 pixels;

The training data contains a list of 7049 images. Each row contains the (x,y) coordinates for 15 keypoints, and image data as row-ordered list of pixels:<br />  
![Alt text]( training_data.png?raw=true "")<br />  

Test data contains a list of 1783 test images. Each row contains ImageId and image data as row-ordered list of pixels:<br />  
![Alt text]( test_data.png?raw=true "")<br />  

### --- CNN Model ---<br />
The formula for calculating the output size(height or length) for any given convolutional layer is<br />
![Alt text]( cnn_layer_compute.jpg?raw=true "")<br />
where O is the output height/length, W is the input height/length, K is the filter size, P is the padding, and S is the stride.<br /><br />

#### The detailed design in CNN model:<br />
![Alt text]( cnn_model.jpg?raw=true "")<br />

The input image in the training set is 96x96 pixels.<br />

The network contains 4 convolutional layer packs and 3 fully connection layers. All kernels for the 4 conv-layers have the same size of 3x3 and the same stride of 1. Each conv-layer is followed by a ReLU activation and a 2x2 max pooling layer with stride of 2.<br />

The first convolutional layer contains 32 filters. The output feature map is sized 94x94. After max pooling, the shape of output feature map is 47x47x32.<br />

The second convolutional layer contains 64 filters. The output feature map is sized 45x45. After max pooling, the shape of output feature map is 23x23x64.<br />

The third convolutional layer contains 128 filters. The output feature map is sized 21x21. After max pooling, the shape of output feature map is 11x11x128.<br />

The fourth convolutional layer contains 256 filters. The output feature map is sized 9x9. After max pooling, the shape of output feature map is 5x5x256.<br />

After the 4 convolutional layer packs, 3 fully connected layers are followed. Each layer has 512, 512 and 30 outputs. A dropout are applied on FC2. The final output vector contains regression value of positions of the 15 keypoints(x, y) <br />

### --- Optimizer ---<br />
Stochastic Gradient Descent (SGD) and its variants are probably the most used optimization algorithms for machine learning in general and for deep learning in particular.<br />
![Alt text]( sgd.jpg?raw=true "")<br />
Adam is a method for efficient stochastic optimization that only requires first-order gradients with little memory requirement. Adam Optimizer is used in this project.

### --- Evaluation ---<br />
Submissions are scored by Kaggle on the Root Mean Squared Error(RMSE):<br />
![Alt text]( rmse.jpg?raw=true "")<br />
The final score of this submission is 3.47.<br />

### --- Result visualization ---<br />
![Alt text]( result.png?raw=true "")<br />
