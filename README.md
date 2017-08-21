# Traffic Sign Recognition

---

## Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[histogram]: ./output_images/histogram.png "histogram of traffic signs"
[16images]: ./output_images/16images.png "16 random images"
[orig]: ./output_images/orig.png "color image"
[gray]: ./output_images/gray.png "gray image"
[5images]: ./output_images/5images.png "5 test images"

## Rubric Points

You're reading it! and here is a link to my [project code](https://github.com/morgengc/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. 

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is (34799, 32, 32, 3)
* The size of the validation set is (4410, 32, 32, 3)
* The size of test set is (12630, 32, 32, 3)
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the datasets are classified into 43 classes.

![alt text][histogram]

And I choosed 16 random images and titled with their labels as following:

![alt text][16images]

### Design and Test a Model Architecture

#### 1. How I preprocessed the image data. 
As a first step, I decided to convert the images to grayscale because in this question we need the image shape information rather than the color depth. I changed the RGB images to gray scale using a function named `rgb2gray()`:
```
def rgb2gray(rgb):
    r, g, b = rgb[:,:,:,0], rgb[:,:,:,1], rgb[:,:,:,2]
    gray = 0.2989*r+0.5870*g+0.1140*b
    return gray
```
Here is an example of a traffic sign image before and after grayscaling.

![alt text][orig] ![alt text][gray]

Then I normalized the image data using the algorithm below, to make the data share a mean zero and equal variance.
```
def process(x):
    gray = rgb2gray(x)
    x_normalized = np.array([(gray[i,:,:]-128.0)/128.0 for i in range(len(gray))]).reshape(gray.shape+(1,))
    return x_normalized
```

#### 2. Model architecture
I used a LeNet architecture with two conv layers, two pooling layers, and three fully connected layers.

Layers description as below:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution     	    | 1x1 stride, same padding, outputs 28x28x6     |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6   				|
| Convolution           | 1x1 stride, same padding, outputs 10x10x16    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Fully connected		| 3 layers. output 400->120->84->43             |
| Dropout               |  0.7       									|

#### 3. Train the model

. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used many hyperparameters, including `EPOCHS = 15`, `BATCH_SIZE = 128`, `rate = 0.002`, and these paramters lead a best result.

#### 4. Hyperparameter tuning
The first hyperparameter be tuned is `rate`. I tested 0.001, 0.0015, 0.002, 0.0025, and I found 0.002 is better in most case.

After that, I changed `EPOCHS`, using 10, 15, 16, 17 and 20. I found in most case, `EPOCHS` should not exceed 15, otherwise the model will be overfitted.

My final model results were:
* training set accuracy of 0.992
* validation set accuracy of 0.937
* test set accuracy of 0.924

Since I choosed a well known model called LeNet, I should answer these following questions:

* What architecture was chosen?
    * LeNet
* Why did you believe it would be relevant to the traffic sign application?
    * LeNet is a good solution to identify a gray scale image into a certain class, people trained this model using MNIST datasets. While traffic sign recognition is a image classification problem too. Although traffic sign is a RGB image, our problem do not care about the color depth, only the shape of object is useful. So we can change color image to gray scale image, and use LeNet to identify the meaning of traffic sign.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    * Since the accuracy on every datasets are near 100%, and every time I got almost the same accuracy, I believe this model is working well.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web
Well, I choosed five traffic signs below, I changed them to *.jpg, and resize all these images to 32x32.

Here are these images:

![alt text][5images]

The forth image might be difficult to classify because it's not in the 43 classes. 

#### 2. Discuss the model's predictions on these new traffic signs

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (60km/h) | Speed limit (60km/h) | 
| Keep right | Keep right |
| Vehicles over 3.5 metric tons prohibited | Vehicles over 3.5 metric tons prohibited |
| Speed limit (40km/h) | Speed limit (60km/h) |
| Slippery road | Road work |

The accurancy of new images is 60%, this compares favorably to the accuracy on the test set of 91.6%. 

Since I choose the first three signs from a German traffic document, they were very standard traffic signs, and were well recongnized. And the latter two signs were downloaded from Internet, the 40km/h speed limit sign is not contained in the 43 known classes, so I can not identify it. The last one seems to be a slippery road, but is a little different from standard slippery road sign, the model traited it as a road work sign.

#### 3. How certain the model is

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a speed limit sign (probability of 0.98). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98         			| Speed limit (60km/h)                          | 
| .01     				| Speed limit (50km/h)                          |
| .00					| Speed limit (80km/h)                          |
| .00	      			| Stop                                          |
| .00				    | Speed limit (30km/h)                          |


For the second image is similar to the first one, the model is relatively sure that this is a keep right sign (probability of 0.99). The top five soft max probabilities were


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Keep right                                    | 
| .00     				| Dangerous curve to the right                  |
| .00					| General caution                               |
| .00	      			| Roundabout mandatory                          |
| .00				    | Road work                                     |

The third image is similar to the two above. The model is relatively sure that this is a prohibited sign (probability of 0.99). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Vehicles over 3.5 metric tons prohibited      | 
| .00     				| No passing                                    |
| .00					| Speed limit (100km/h)                         |
| .00	      			| End of all speed and passing limits           |
| .00				    | End of no passing by vehicles over 3.5 metric tons |

The forth image is a 40km/h speed limit sign, which is not contained in the known 43 classes. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .57         			| Speed limit (60km/h)                          | 
| .43     				| Speed limit (80km/h)                          |
| .00					| Ahead only                                    |
| .00	      			| No vehicles                                   |
| .00				    | Speed limit (50km/h)                          |

The last image is a slippery road, but is a little different from standard slippery road sign, and the model traited it as a road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .39         			| Road work                                     | 
| .27     				| Bumpy road                                    |
| .20					| Traffic signals                               |
| .07	      			| Dangerous curve to the right                  |
| .02				    | General caution                               |
