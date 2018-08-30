--- Objective ---

The objective of this task is using Convolutional Neural Network(CNN) to predict keypoint positions on face images.

--- Data description ---

The data is from Kaggle, which contains a set of face images. The input image consists of a list of 96*96 pixels. 
https://www.kaggle.com/c/facial-keypoints-detection

Each predicted keypoint is specified by an (x,y) real-valued pair in the space of pixel indices. 15 keypoints in the dataset represent the different elements of the face (left_eye_center, right_eye_center, nose_tip …);

The input image is given in the last field of the data files, consists of a list of pixels, as integers in (0,255). The images are 96x96 pixels;

The training data contains a list of 7049 images. Each row contains the (x,y) coordinates for 15 keypoints, and image data as row-ordered list of pixels;

Test data contains a list of 1783 test images. Each row contains ImageId and image data as row-ordered list of pixels.

--- CNN Model ---<br />

Input (96*96)<br />
Convolutional 32@94*94 -> ReLU -> MaxPool 32@47*47  (3*3 kernel, 2*2 pooling)<br />
Convolutional 64@46*46 -> ReLU -> MaxPool 46@23*23  (2*2 kernel, 2*2 pooling)<br />
Convolutional 128@22*22 -> ReLU -> MaxPool 128@11*11  (2*2 kernel, 2*2 pooling)<br />
Densely connected hidden layers with 500 neurons<br />
Densely connected hidden layers with 500 neurons<br />
Output (30)<br />

--- Evaluation ---<br />
Submissions are scored on the Root Mean Squared Error(RMSE)<br />

--- Visualization ---<br />
![Alt text]( result.png?raw=true "")<br />
