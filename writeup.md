# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./hist.jpg "Visualization"
[image4]: ./data/1.png "Traffic Sign 1"
[image5]: ./data/2.png "Traffic Sign 2"
[image6]: ./data/3.png "Traffic Sign 3"
[image7]: ./data/4.png "Traffic Sign 4"
[image8]: ./data/5.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of evaluation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributed:

![alt text][image1]

### Design and Test a Model Architecture


As a first step, I decided to normalize the images by subtracting 128 from all pixels and dividing all pixels by 128.
In addition I used Image Augementation as mentioned as recommondation from the Review. For augmentation I use a function I found online: https://github.com/vxy10/ImageAugmentation/blob/master/img_transform_NB.ipynb
The transform_image function uses Affine Transformation to rotate the image. Therefore the the classifier gets more robust to overfitting.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout  | Keep 80% |
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 5x5x16       									|
| RELU					|												|
| Dropout  | Keep 80% |
| Max pooling	      	| 1x2 stride,  outputs 5x5x16 				|
| Fully connected		| truncated normal, outputs 120        									|
| RELU					|												|
| Dropout  | Keep 80% |
| Fully connected		| truncated normal, outputs 84        									|
| RELU					|												|
| Dropout  | Keep 80% |
| Fully connected		| truncated normal, outputs 43        									|
| RELU					|												|
| Dropout  | Keep 80% |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 20 epoches, a batch size of 128 and a learning rate of 0.01.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy 1st run: 99.7 % second run: 98.5 %
* validation set accuracy 1st run: 94.6 % second second: 93.3 %
* test set accuracy 1st run: 90.2 %, second run: 86.4 %

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
First I started with the LeNet architecture. Therefore I had to convert my images to grayscale. I used this architecture because I think it is a good starting point.
* What were some problems with the initial architecture?
I tried to tune the parameters, but the results were unsatisfing and the model seemed to overfit. I added image augmentation to get more robust.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Later I changed the LeNet Architecture to accept color images as the color is an important information for detecting the correct traffic sign.
In addition I added dropouts and image augmentation to get more robust against overfitting.
* Which parameters were tuned? How were they adjusted and why?
I reduced the learning rate to avoid early overfitting and increased the number of epoches to get better results.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work     		| Road work   									| 
| Priority road     			| Priority road 										|
| Turn left ahead					| Turn left ahead											|
| 60 km/h	      		| Keep right				 				|
| Yield			| Yield     							|


The model was able to detect 5 out of 5 traffic sign correct.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

On all images the model is extremly uncertain (below 20 %). The images seem to be extrem different to the ones it was trained on. The softmax details for each sign are in the ipynb file under "Step 3"
