## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./examples/example_output.jpg)

Overview
---
When we drive, we use our eyes to decide where to go. The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle. Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

[//]: # (Image References)

[image1]: ./output_images/road_img.jpg "Road Image"
[image2]: ./output_images/camera_cal_distorted.jpg "Chessboard Image"
[image3]: ./output_images/camera_cal_undistorted.jpg "Chessboard Image undistorted"
[image4]: ./output_images/camera_cal_undistortedandwarped.jpg "Chessboard Image undistorted and warped"
[image5]: ./output_images/road_undistorted.jpg "Road Image undistorted"
[image6]: ./output_images/road_img_warped.jpg "Road Image warped"
[image7]: ./output_images/r_channel.jpg "R Channel"
[image8]: ./output_images/s_channel.jpg "S Channel"
[image9]: ./output_images/r_binary.jpg "R Binary"
[image10]: ./output_images/s_binary.jpg "S Binary"
[image11]: ./output_images/sxbinary.jpg "Sx Binary"
[image12]: ./output_images/color_binary.jpg "Color Binary"
[image13]: ./output_images/binary_warped.jpg "Combined Binary"
[image14]: ./output_images/slidingwindow.jpg "Sliding Window"
[image15]: ./output_images/result.jpg "Result"




Repository contents
---

**Project.ipynb** contains a Jupyter notebook with the project code and comments to be run in a browser

**AdvancedFindingLanes.py** contains the same project code to be run on a local machine

**camera_cal** folder containing chessboard images for camera calibration

**test_images** folder containing test images

**test_videos** folder containing test test_videos

**output_images** folder containing output images

**project_video** containing the output of the video processing pipeline



The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


Camera calibration
---

All cameras are prone to distortion. In order to perform a lane finding algorithm on camera images, the camera needs to be calibrated first. This can be done by chessboard calibration.

The code for this step is contained in the second code cell of the IPython notebook located in "./Advanced Lane Detection.ipynb"

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![Chessboard Image][image2]
![Chessboard Image Undistorted][image3]
![Chessboard Image Undistorted and Warped][image4]


###Image Processing pipeline

Distortion Correction
---

Utilizing the camera matrix mtx and distortion coefficients dist from the camera calibration step, I can now undistort any images taken by the camera. This is done by calling the `cv2.undistort()` function on the image

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![Road Image][image1]
![Road Image Undistorted][image5]

Color and Gradient Thresholding
---

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in cells 8 and 9). Utilized in this step were the S-channel(from HLS space), the R-channel(from RGB space) and the sobel gradient in x-direction. Threshold ranges of min = 115, max = 255 for S, min = 10, max = 150 for Sx and min = 190, max = 255 for R have proven to bring reasonable results.
I can tehn use these binaries to create a combined binary and stacked color binary.


Here's an example of my output for this step.  

![R Channel][image7]
![S Channel][image8]
![R Binary][image9]
![S Binary][image10]
![Sx Binary][image11]
![Colored Binary][image12]
![Combined Binary][image13]

Perspective Transformation
---

The code for my perspective transform includes a function called `perspectieTransform()`, which appears in the 6th and 7th code cell of the IPython notebook).  The `perspectiveTransform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[240, 700], [580, 460], [705, 460], [1085, 700]])
dst = np.float32([[250, 700], [250, 0], [1000, 0], [1000, 700]])
```
The source points form a trapezoid shape on the lane-lines of the road image. The perspective is then warped from a front facing view to a birds-eye-view. The lane-lines now appear as parallel lines.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Road Image warped][image6]

Identifying Lane Pixels
---

Now that I have a warped, binary image of the lane-lines I can perform the actual lane detection. This is done by executing a sliding window algorithm. First I take a 0-axis histogram of the bottom half of the image. The two peaks of this histogram are at the x-position of the lanes closest to the vehicle (since the lines are almost vertical). These x-position are the starting point of the sliding window search. I then iterate through a total of 9 windows, counting the number of pixels in the window. If the number of pixels is higher than a margin, the window is shifted.
In a second step, the pixels found in the search are then fitted with second order polynomials. In the end I get the polynomial coefficents for both lane lines and their x and y values for plotting.

Once a pair of polynomials is fitted to the lanes a quicker search algorithm can be used in the following iteration. The search-from-prior algorithm searches for lane pixels within a set margin from the last lane-line. If no pixels can be found, sliding window search is executed.

![Sliding Windows][image14]

Curvature and Offset
---

Given the x and y locations of the lane-lines we can now calculate the curvature using the following formular. First we have to convert from pixel space to meters by multiplying by a correction factor.

ym_per_pix = 30/720
xm_per_pix = 3.7/700

The curvature is calcuated for the y-location closest to the vehicle.

Another parameter important for performing lane centering control is the offset, or crosstrack error.
Since the camera is mounted at the center of our car, the center of the image represents the center of the vehicle frame. The offset can easily be calculated by subtracting the lane center from the vehicle center. (Again values need to be converted from pixel space to meters first.)

Result
---

![Result][image15]

With the Minv matrix from the earlier perspective transformation step we can now transform the image back and project the detected lane-lines onto the image.
A second function adds the curvature and offset measurements to the image.
