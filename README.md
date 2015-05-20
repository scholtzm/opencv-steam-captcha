# Steam Captcha Solver using OpenCV

![alt tag](https://raw.githubusercontent.com/scholtzm/opencv-steam-captcha/master/data/2GCZ4A.png)

This is an experimental/learning project which attemps to solve the captcha used at steamcommunity.com. This project is currently just a proof of concept and *does not* solve the captcha completely.

### Data

The data set consists of 100 manually labeled images. Each image contains exactly 6 characters.

By inspecting the images further, we can see that the character set consists of letters "A-Z", numbers "0-9" and special characters "@%&". However, several characters have been left out, specifically "0156OIS".

This gives us: (100 \* 6) / 32 = **18.75** samples per character

This type of captcha is no longer used but it's still available at the following address:

`https://steamcommunity.com/tradeoffer/new/captcha?v=A&sessionid=B&partner=C`

* A is a random timestamp (number).
* B is legitimate `sessionid` which matches the cookie value.
* C is partner's ID.

### Algorithm

The algorithm consists of 3 parts:

1. Thresholding - the threshold value is calculated dynamically and it's based on the images histogram.
2. Segmentation - each letter is cropped out of the image and stored separately.
3. Classification - very simple classification which uses SVM and HoG (both available in OpenCV).

### Results

Segmentation of the letters works quite well and 85 out of 100 images from the data set were fully processed. If we look at the individual characters, the segmentation algorithm worked for 95% of all 600 characters.

Classification is pretty simple and at this point the SVM is able to identify letters G and Y pretty accurately after only using 50 training samples. Much larger data set would be necessary to properly identify all 32 types of characters.

### Build and run

1. Install Boost and OpenCV. Use your favourite package manager for that.
2. Clone this repo.
3. Check the code and slightly modify the paths or anything else you need.
4. Build with cmake.
