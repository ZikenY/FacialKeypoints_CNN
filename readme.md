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
![Alt text]( cnn_model.jpg?raw=true "")<br />

### --- Evaluation ---<br />
Submissions are scored on the Root Mean Squared Error(RMSE)<br />
![Alt text]( rmse.jpg?raw=true "")<br />

### --- Visualization ---<br />
![Alt text]( result.png?raw=true "")<br />
