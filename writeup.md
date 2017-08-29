#**Traffic Sign Recognition** 

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

[image1]: ./report/1.png "Visualization"
[image2]: ./report/sign1.png "Sign 1"
[image3]: ./report/sign2.png "Sign 2"
[image4]: ./report/sign3.png "Sign 3"
[image5]: ./report/sign4.png "Sign 4"
[image6]: ./report/sign5.png "Sign 5"
[image7]: ./report/probabilities.png "Sign probabilities"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32)
Number of classes = 43

####2. Include an exploratory visualization of the dataset.

Here is an example image from the train dataset

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided not to convert the signs to grayscale, since there is information in the color of the sign. Speed signs are red, turn signs are blue, etc.
The only pre-processing I did was dividing by 256 to normalize the images between 0 and 1.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5	    | 14x14x6, outputs 10x10x16 					|
| RELU					|        										|
| Max Pooling			| 5x5x16       									|
| Fully connected		| input 400, output 400     					|
| Fully connected		| input 400, output 200       					|
| Fully connected		| input 200, output 120        					|
| Fully connected		| input 120, output 43       					|
|						|												|
|						|												|
 
I increased the number of fully connected layers, since this architecture was used for mnist, but now we have more classes.
One thing I noticed doing this report is that information is lost in the convolutional layers. Usually when resolution is reduced, the number of images kept is
increased, which did not happen in this case.

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Once I increased the number of layers, the model would keep improving beyond 10 epochs, so I increased it to 50. The batch size was as large as I could fit into my GPU
so 128.
I kept the learning rate at 0.001 and the optimizer was Adam

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.935
* test set accuracy of 0.89

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The initial architecture was LeNet because it was asked in the project. A better architecture might have been vgg16, but maybe the images were too small for so many conv layers
* What were some problems with the initial architecture?
The initial architecture had relatively small layers, considering that we have 43 classes
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I added one fully connected layer and also increased the width of the fully connected layers
* Which parameters were tuned? How were they adjusted and why?
Number of epochs, because more layers were more difficult to train and would keep improving longer
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Convolutional layers work well with this problem because they work well with structured data like images. They help detect patterns anywhere in the image which is useful when the signs are not perfectly centered.
Dropout might have helped to avoid overtfitting, since the network loses many activations and allows it to develop more robust ones.
I tried adding batch normalization to speed up the training but the results were not much better.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]

All the images were high quality and easy to classify. The only one more challenging was the 60 speed limit, because it is easily confused with speed limits for other speed
This one was indeed the only failed image, because it recognised it as a speed limit of 30. The resulting accuracy was 0.8

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep right      		| Keep right 									| 
| General warning  		| General warning								|
| 60km/h				| 30km/h										|
| STOP     				| STOP					 						|
| Turn right ahead		| Turn right ahead	 							|


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

Next are the images ordered by how confident the model was of their detection. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 163         			| Keep right 									| 
| 94     				| General warning 								|
| 45					| Turn right ahead								|
| 20	      			| Stop							 				|
| 14				    | 60km/h   										|

In this image, you can see all the images and their probabilities.

![alt text][image7]

####How does the accuracy in the new images compare to the model's accuracy in the original test set?
It's difficult to say because of the low number of samples. The accuracy of the test set was 0.89 and the accuracy of the new images was 0.8.
While they were new images, they also were very high quality pictures, compared to the more realistic images of the test set.
I think in general, though a 0.8 means the model has generalized well, and the simple images work well. The only image with problems was the one with the speed sign, which presumably already had problems in the original datasets.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


