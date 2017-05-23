# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_img/car_noncar.png
[image2]: ./writeup_img/HOG_example.png
[image3]: ./writeup_img/one_scale.png
[image4]: ./writeup_img/two_scales.png
[image6]: ./writeup_img/labels_map.png
[image7]: ./writeup_img/output_bboxes.png
[image8]: ./writeup_img/heatmap1.png
[image9]: ./writeup_img/heatmap2.png
[image10]: ./writeup_img/heatmap3.png



### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 6 code cell of the IPython notebook in function `get_hog_features()`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I found out that starting HOG parameters `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` are pretty good to help detect cars.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using spatial features, histogram features and HOG features with params:
`orient = 9`
`pix_per_cell = 8`
`cell_per_block = 2`
`spatial_size=(32, 32)`
`hist_bins = 32`
`hog_channel = "ALL"`

The code for this step is contained in the 6 code cell of the IPython notebook in function `train_classifier()`. The HOG features extracted from the training data have been used to train a SVM linear kernel classifier. Features should be scaled to zero mean and unit variance before training the classifier.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search window positions from `ystart = 400` to `ystop = 656`. This allows the vehicle detector to perform faster than searching through the entire image, and also reduces potential false positives in areas like the sky and trees.

I chose `window_size=(64,64)` at `scale = [1.5]` and an overlap percentage of 0.75 (i.e. 75%). Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance. This means that a cells_per_step = 2 would result in a search window overlap of 75%

I found this gave me reliable detections with multiple detected bounding boxes on vehicles, i.e. the density of detections around vehicles is high, relative to non-vehicle areas of the image. This allows me to create a high-confidence prediction of vehicles in my heat map (more details later) and reject potential false positives.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scale `scale = [1.0, 1.5]` using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example image:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
[![link to my video result](https://img.youtube.com/vi/GAzoAtJP110/0.jpg)](https://youtu.be/GAzoAtJP110)

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are three frames and their corresponding heatmaps:

![alt text][image8]
![alt text][image9]
![alt text][image10]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all three frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

The main challenge for this project was parameter tuning, mostly to reduce the number of false positives in the video. Even though the HOG+SVM classifier reported good results after training, it did not necessarily mean good results in the overall vehicle detection task.

It took me some time to figure out what size of scaling windows to use, and how many of them. When I increased number of windows, I had to tune up magnitude of threshold. Finally I ended up with two windows at scale 1.0 and 1.5. I don't separete them to use smaller one on upper side of image but it could increase speed of the detector, especially when lane detection is done at the same time!

One final potential areas were the current vehicle detection pipeline could fail is when two cars are very close to each other. Sometimes model treated them as one big vehicle.