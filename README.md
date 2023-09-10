# Vision and Perception - Final Project

-------------------- By Giulia Nardini, Charlotte Ludovica Primiceri, Carmine Zito --------------------------

## Introduction
For our final project we have focused on the OBJECT DETECTION TASK performed on videos in the Cityscapes Dataset.

Our aim was to investigate wheter the segmentation task could be improved or not training models on the depth maps and the optical flow graphic representations in addiction to original images.

We have performed image segmentation on a video taken from Cityscapes Dataset using three different models:
  1)  Mask-RCNN pre-trained on COCO Dataset (Detectron2 implementation)
  2)  Pytorch Unet with double input channel
  3)  Pytorch Unet with single input channel (as original, that we trained from start)

## Models and Experiments
### Mask-RCNN

![stuttgart_02_000000_005395_leftImg8bit](https://github.com/CharlottePrimiceri/VP_Project/assets/114931709/98b510ac-ecb1-4775-89a4-a30555609e85)

![stuttgart_02_000000_005813_leftImg8bit](https://github.com/CharlottePrimiceri/VP_Project/assets/114931709/81571945-8670-43ef-9c4b-1ae4fded5c1d)

#### Cityscapes to COCO conversion
We tried to perform fine-tuning of the Mask-RCNN on the cityscapes dataset and we transformed our dataset in COCO format.
We didn't manage to re-train the Mask-RCNN due to lack of computational resources and the complexity of the model (...), so at the end we opted for the Pytorch Unet, a simpler model on which we made more experiments.
### Pytorch Unet 1 channel
Target example:

![image](https://github.com/CharlottePrimiceri/VP_Project/assets/114931709/01be8415-b49e-44ce-b896-53868e6ba2f2)

We trained from start the orginal Pytorch Unet for 15 epochs and the result we got was not so distant from the target. 

Some examples from our predictions:

![image](https://github.com/CharlottePrimiceri/VP_Project/assets/114931709/2f3b7dfd-cba0-4dc0-a55b-1e4f983ef106)


### Pytorch Unet 2 channels
We trained this alternative network for 15 epochs and the same training details as before.

Unfortunally, this is what we obtained:

![image](https://github.com/CharlottePrimiceri/VP_Project/assets/114931709/30989a00-fe73-4577-ac70-67adc91b7bb0)


Due to the fact that the loss at the end of the training is stabilized at a value of about 0.4, training the model further wouldn't be useful. Training the model for more epochs has lead us to predict all black images, as if the model had found how to minimize the loss in the easiest way possible.
After having taken some experiments, we can't conclude that giving to the model the "hint" of the depth map helps the model predicting the segmentated image.

## Disparity Map and Depth Map

Initially we worked on Disparity Map method in order to estimate distance of detected objects, because of its reliablity and accuracy to identify the shapes. 
But the model required a pair of images for each frame (one frome the "left eye" and the other for the "right eye"), this issue made the model unsuitable for the Unet, that is trained with a dataset composed by "monocular" images.
Beacuse of this we moved on to a more suitable method so we chose MiDaS model, that computes depth map starting from a single image.
Even if MiDaS provides a low quality map compared to the Disparity Map, it comes with better scalability that made the combination between Unet mask and distance map very easy.

## Depth Map

![image](https://github.com/CharlottePrimiceri/VP_Project/assets/114931709/c562fef0-d3c7-4729-9c0d-37f968e20715)

Different ways tried out: .... show results


## Optical flow

To compute the optical flow at the beginning we were thinking of applying a pre-trained model, then we implemented the Lucas-Kanade algorithm.

Lucas-Kanade: 

![image](https://github.com/CharlottePrimiceri/VP_Project/assets/114931709/18f40f4e-1320-4093-a297-e269d041c165)

![image](https://github.com/CharlottePrimiceri/VP_Project/assets/114931709/8e937b50-5b61-4ee2-ae30-7756f9fcd055)

![image](https://github.com/CharlottePrimiceri/VP_Project/assets/114931709/1b2c4fa3-682e-4213-91ad-b47a7beb1a8a)


## Showing results

TO HAVE A LOOK AT OUR FINAL RESULTS ON VIDEO SEGMENTATION PLEASE GO TO segmented_video FOLDER!!!

## References:
- Cityscapes dataset: https://www.cityscapes-dataset.com
- Cityscapes to COCO conversion: https://github.com/mcordts/cityscapesScripts
- Mask-RCNN: https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf
- Mask-RCNN with disparity map and optical flow: https://github.com/thiagortk/maskRCNN-DisparityMap-OpicalFlow
- Detectron2: https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
- U-Net: https://github.com/goldbattle/pytorch_unet/tree/master
- Optical Flow Trajectories: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
- MiDas: https://pytorch.org/hub/intelisl_midas_v2/ 


