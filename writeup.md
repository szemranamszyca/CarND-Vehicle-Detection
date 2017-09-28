# **Vehicle Detection Project**

## Arkadiusz Konior - Project 5.

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car1]:  ./imgs/car1.png
[non1]: ./imgs/non1.png
[hogcar1]: ./imgs/hog1car.png
[hognon1]: ./imgs/hog1non.png


[scale15]: ./imgs/scale15.png
[scale2]: ./imgs/scale2.png
[scale25]: ./imgs/scale25.png
[test]: ./imgs/finish_test.png
[heatmap]: ./imgs/heat_example.png

[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points 

---


### Histogram of Oriented Gradients (HOG)

#### 1. HOG features from the training images.

Code for extracting features could be found at *file feature_extract.py* and *pipeline.py*

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Car example][car1]
![Non-car example][non1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![HOG Car example][hogcar1]
![HOG Non-car example][hognon1]

#### 2. HOG parameters.

I've extracted features from images with parameters:

+ color_space='YCrCb'
+ spatial_size=(32, 32
+ hist_bins=32
+ orient=9
+ pix_per_cell=8
+ cell_per_block=2

Features were split to train-test set (80%/20%) and normalize. I've decided to train SVM linear classifier, and these parameters gave me more than 98% accuracy. To speed-up further work, I've save model and scaler to file using joblib functions.

### Sliding Window Search

#### 1. Sliding window search

Final sliding window search could be find at *find_cars.py* and it is used at *process_video.py*. I've decided to searched on three scales using YCrCb 3-channel HOG features.

```python3
ystart_ystop_scale = [(350, 550, 1.5), (400, 620, 2), (440, 700, 2.5)]
```

![Scale 1.5][scale15]
![Scale 2][scale2]
![Scale 2.5][scale2.5]

Here's example of final result on test image:

![Final result][test]
---

### Video Implementation

####  1. Link to your final video output. 
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

I've created HeatingControl class, where I kept "memory" of ten last frames. After each 10 frames, all fields of class are set to inital state.

Here's example

![Heatmap example][heatmap]

---

### Discussion

One the video, sometimes false positive windows are visible, especially on the left side of the road. Despite that, nearest cars are always detected correctly. Tuning model parameters might help with that issue.

Slide window alghorithm had quite poor performance (2frames/second). I thought, that extending my HeatingControl class might be solution for that. I could implement part for "cooling down" and "heating up" part of images and searching cars only in these areas. 

