## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<a href="http://www.youtube.com/watch?feature=player_embedded&v=zwTB2HMuP6Y" target="_blank"><img src="http://img.youtube.com/vi/zwTB2HMuP6Y/0.jpg" 
alt="This My test video" width="960" height="540" border="10" /></a>

### This My test video

## In this project, my goal is to write a software pipeline to identify the lane boundaries in a video.  

### Overview
---

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

[//]: # (Image References)

[image1]: ./readmemd/Compute_the_camera_result.png "Compute_the_camera_result"
[image2]: ./readmemd/Distortion.png "Distortion"
[image3]: ./readmemd/Color_result.png "Color_result"
[image4]: ./readmemd/Color_HLS_S_threshold_result.png "Color_HLS_S_threshold_result"
[image5]: ./readmemd/Color_result.png "hough_image_solidYellowLeft"
[image6]: ./test_images_output/solidYellowLeft.jpg "solidYellowLeft"

The Project
---

The goals / steps of this project are the following:

* Camera Calibration
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

My pipeline consisted of some steps.

* 01.Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* 02.Apply a distortion correction to raw images.
* 03.Use color transforms, gradients, etc., to create a thresholded binary image.
* 04.Use to correctly rectify each image to a "birds-eye view"
* 05. Identified lane-line pixels and fit their positions
* 06.Determine the curvature of the lane and vehicle position with respect to center
* 07.Warp the detected lane boundaries back onto the original image.
* 08.Pipeline_video


#### 01.Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

In this stage, I read in some test images and processed the camera corrections according to the way the course taught. However, some images were found to be unrecognizable during the calibration process, so the Internet searched for relevant data and found that the input image did not necessarily match the desired image. So I tried to fine-tune Nx, Ny up and down, still can't read two pictures. So I chose to exclude two images and use other images as the correction pattern.

```python
#Tried to fine-tune Nx, Ny up and down
if ('calibration1.jpg' in image_name):
    nx = 9
    ny = 5
else:
    nx = 9
    ny = 6

```
Result
![alt text][image1]


#### 02.Apply a distortion correction to raw images.

After camera correction, I saved the parameters in PKL form. The parameters are then referenced in this stage and applied to each image.

```python
import pickle

f = open('Camera_parameters.pkl', 'wb')
pickle.dump(mtx, f)
pickle.dump(dist, f)
f.close()

```

Result
![alt text][image2]

#### 03.Use color transforms, gradients, etc., to create a thresholded binary image.

In this stage,I tried to use the various color leathers I found in the documentation to transform the image and find a workable solution in it.

```python
for image_name in os.listdir("output_images/Undistorted_Image/"):
    #read in each image
    image = mpimg.imread("output_images/Undistorted_Image/" + image_name)
    #Show Color
    fig, axs = plt.subplots(4,3, figsize=(20, 20))
    axs = axs.ravel()

    axs[0].imshow(image[:,:,0], cmap='gray')
    axs[0].set_title('R-channel', fontsize=30)
    axs[1].imshow(image[:,:,1], cmap='gray')
    axs[1].set_title('G-Channel', fontsize=30)
    axs[2].imshow(image[:,:,2], cmap='gray')
    axs[2].set_title('B-channel', fontsize=30)

    image_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    axs[3].imshow(image_HSV[:,:,0], cmap='gray')
    axs[3].set_title('H-Channel', fontsize=30)
    axs[4].imshow(image_HSV[:,:,1], cmap='gray')
    axs[4].set_title('S-channel', fontsize=30)
    axs[5].imshow(image_HSV[:,:,2], cmap='gray')
    axs[5].set_title('V-Channel', fontsize=30)

    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    axs[6].imshow(image_HLS[:,:,0], cmap='gray')
    axs[6].set_title('H-Channel', fontsize=30)
    axs[7].imshow(image_HLS[:,:,1], cmap='gray')
    axs[7].set_title('L-channel', fontsize=30)
    axs[8].imshow(image_HLS[:,:,2], cmap='gray')
    axs[8].set_title('S-Channel', fontsize=30)
    
    image_LAB = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    axs[9].imshow(image_LAB[:,:,0], cmap='gray')
    axs[9].set_title('L-Channel', fontsize=30)
    axs[10].imshow(image_LAB[:,:,1], cmap='gray')
    axs[10].set_title('A-channel', fontsize=30)
    axs[11].imshow(image_LAB[:,:,2], cmap='gray')
    axs[11].set_title('B-Channel', fontsize=30)

```
Result
![alt text][image3]

After finding the color, I try to find the upper and lower thresholds as filters.

Define function
```python
def HLS_S_threshold(img, threshold=(200, 255)):

    HLS_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    HLS_S_img = HLS_img[:,:,2]
    
    HLS_S_img = HLS_S_img*(255/np.max(HLS_S_img))
    
    binary_output = np.zeros_like(HLS_S_img)
    
    binary_output[(HLS_S_img > threshold[0]) & (HLS_S_img <= threshold[1])] = 1

    return binary_output

```
Show Result
```python
for image_name in os.listdir("output_images/Undistorted_Image/"):
    #read in each image
    image = mpimg.imread("output_images/Undistorted_Image/" + image_name)
    
    #It is best to test the threshold of that section.
    #Show Result
    fig, axs = plt.subplots(1,3, figsize=(20, 9))
    axs = axs.ravel()
    axs[0].imshow(HLS_S_threshold(image, threshold=(0, 80)))
    axs[0].set_title('HLS_S-0-80-Channel', fontsize=30)
    axs[1].imshow(HLS_S_threshold(image, threshold=(80, 160)))
    axs[1].set_title('HLS_S-80-160-Channel', fontsize=30)
    axs[2].imshow(HLS_S_threshold(image, threshold=(160, 255)))
    axs[2].set_title('HLS_S-160-255-Channel', fontsize=30)
    
    
```
Result
![alt text][image4]

After testing, you can know that 160 is the most suitable.
