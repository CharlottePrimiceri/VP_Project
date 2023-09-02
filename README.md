# Vision and Perception - Final Project

-------------------- By Giulia Nardini, Charlotte Ludovica Primiceri, Carmine Zito --------------------------

## Introduction
For our final project we have focused on the OBJECT DETECTION TASK performed on the Cityscapes Dataset.

Our aim was to investigate wheter the segmentation task could be improved or not training models on the depth maps and the optical flow graphic representations in addiction to original images.

We have performed image segmentation on a video taken from Cityscapes Dataset using three different models:
  1)  Mask-RCNN pre-trained on COCO Dataset (Detectron2 implementation)
  2)  Pytorch Unet with double input channel
  3)  Pytorch Unet with single input channel (as original, that we trained from start)

## Models and Experiments
### Mask-RCNN

#### Cityscapes to COCO conversion
We tried to perform fine-tuning of the Mask-RCNN on the cityscapes dataset and we transformed our dataset in COCO format.
We didn't manage to re-train the Mask-RCNN due to lack of computational resources and the complexity of the model (...), so at the end we opted for the Pytorch Unet, a simpler model on which we made more experiments.
### Pytorch Unet 1 channel
Target example:

![image](https://github.com/CharlottePrimiceri/VP_Project/assets/114931709/01be8415-b49e-44ce-b896-53868e6ba2f2)

We trained from start the orginal Pytorch Unet for 15 epochs and the result we got was not so distant from the target. 

An example from our result:


### Pytorch Unet 2 channels
We trained this alternative network for 15 epochs and the same training details as before.
Unfortunally, this is what we obtained:

Due to the fact that the loss at the end of the training is stabilized at a value of about 0.4, training the model further wouldn't be useful. Training the model for more epochs has lead us to predict all black images, as if the model had found how to minimize the loss in the easiest way possible.
After having taken some experiments, we can't conclude that giving to the model the "hint" of the depth map helps the model predicting the segmentated image.


## Depth Map

![image](https://github.com/CharlottePrimiceri/VP_Project/assets/114931709/c562fef0-d3c7-4729-9c0d-37f968e20715)

Different ways tried out: .... show results


## Optical flow

To compute the optical flow we implemented the Lucas-Kanade algorithm and compared it with the output obtained with the FlowNet.

## Showing results


## References:
- Cityscapes dataset: https://www.cityscapes-dataset.com
- Cityscapes to COCO conversion: 
- Mask-RCNN: https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf
- Mask-RCNN with disparity map and optical flow: https://github.com/thiagortk/maskRCNN-DisparityMap-OpicalFlow
- Detecron2: https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
- FlowNet: 
  


