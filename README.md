#                                                   Vision and Perception - Final Project

-------------------------------- By Giulia Nardini, Charlotte Ludovica Primiceri, Carmine Zito -----------------------------------------

## Checkpoints
In this folder you can find our checkpoints for single and double channel UNets (they are to big to upload here).

https://drive.google.com/drive/folders/11pa-SYHlBT2aytHaOC5VzxrfroDWzgCS?usp=sharing

## Introduction
For our final project we have focused on the OBJECT DETECTION TASK performed on videos in the Cityscapes Dataset.

Our aim was to investigate whether the segmentation task could be improved or not by training deep learning models on the depth maps. At the beginning we tought about enforcing the prediction of the segmented images through the graphic representation of the optical flow, but unfortunately this was far beyond our possibilities. Anyway, we applied an algorithm in order to plot the trajectories of segmented objects in videos based on the optical flow computation (which we implemented also "by hand").

We have performed image segmentation on a video taken from Cityscapes Dataset using three different models:
  1)  Mask-RCNN pre-trained on COCO Dataset (Detectron2 implementation);
  2)  Pytorch Unet with double input channel trained from start;
  3)  Pytorch Unet with single input channel re-trained from start for comparisons with 2).

## Models and Experiments
### Mask-RCNN 

![stuttgart_02_000000_005395_leftImg8bit](https://github.com/CharlottePrimiceri/VP_Project/assets/114931709/98b510ac-ecb1-4775-89a4-a30555609e85)

![stuttgart_02_000000_005813_leftImg8bit](https://github.com/CharlottePrimiceri/VP_Project/assets/114931709/81571945-8670-43ef-9c4b-1ae4fded5c1d)

#### Cityscapes to COCO conversion
We tried to perform fine-tuning of the Mask-RCNN on the Cityscapes Dataset and we transformed our dataset annotations in COCO format because this is the requested input for this model in Detectron2.

#### Unable to fine-tune on Cityscapes Dataset
We didn't manage to re-train the Mask-RCNN due to lack of computational resources and the complexity of the model (...), so at the end we opted for the Pytorch Unet, a simpler model on which we were able to make more experiments. We also realized that it makes no sense training a model that has already reached very high performances.

#### Model settings
Because we were interested in classifying and segmenting only three classes (pedestrians, cars, bicycles) we filtered our annotations to focus only on these categories. Moreover, in order to show a better visualization of the output predictions we set the segmentation color as shades of the same tone for each category. As another pre-processing step we had to register our Costum Dataset (Cityscapes with filtered COCO annotations).
We specify that we worked with a 'cpu' accelerator to make predictions. 

### Single-channel Pytorch U-Net
Target example:

![image](https://github.com/CharlottePrimiceri/VP_Project/assets/114931709/01be8415-b49e-44ce-b896-53868e6ba2f2)

We trained from start the orginal Pytorch Unet for 50 epochs and the result we got was not so distant from the target. 

Some examples from our predictions with 50 epochs training:

![image](https://github.com/CharlottePrimiceri/VP_Project/assets/114931709/2f3b7dfd-cba0-4dc0-a55b-1e4f983ef106)

Examples of predicted batch with 20 epochs training:

############ add image 

#### Training details:
- batch_size: 16
- learning_rate: 0.0002
- epochs: 20, 50
- accelerator: 'gpu'

#### Train loss curve:

![loss_curve_nodepth_50_epochs](https://github.com/CharlottePrimiceri/VP_Project/assets/114931709/e46281c2-8d04-48b7-a507-f0c126832c14)

(this graph refers to 50 epochs training!)

From the graph it is clear that the curve has saturated. Having not printed the validation loss curve, in order to avoid incurring in the overfitting problem, we preferred to stop the training at 20 epochs. Going on training this network until 50 epochs we have noticed that the loss hasn't't decreased significantly since epoch 20.

#### Model evaluation: 

![image](https://github.com/CharlottePrimiceri/VP_Project/assets/114931709/0babd4ef-76f5-4336-83bd-55320277c3ec)


##### accuracy with 50 epochs:

![image](https://github.com/CharlottePrimiceri/VP_Project/assets/114931709/1178f7e8-7f61-41c6-abf5-a1fc170d94f0)


##### accuracy with 20 epochs:

![image](https://github.com/CharlottePrimiceri/VP_Project/assets/114931709/f0bdd6fd-1064-4c51-8282-6e2cca4929b5)


### Double channel Pytorch U-Net
At the beginning we trained this alternative network for **20 epochs**, but, unfortunately, running the predict code we obtained all black images. This result is a clear sign of overfitting! The network had learned trick of how to minimize the loss with minimum effort: generating all equal black images. So, we decided to stop the training at **epoch 4** and the results were much weaker than those obtained with the other unet. Anyway, it is clear that with 4 epochs the network is able to distinguish at least the roads.

This is the segmented image we obtained loding model's weights from checkpoint at epoch 4:

![image](https://github.com/CharlottePrimiceri/VP_Project/assets/114931709/0c536a6f-b045-43be-a1ad-4ddf4224d652)


############### change comment after printing the loss
Due to the fact that the loss at the end of the training is stabilized at a value of about 0.4, training the model further wouldn't be useful. 
After having taken some experiments, we **can't conclude that giving to the model the "hint" of the depth map helps the model predicting the segmentated image**.

#### Training details:
- batch_size: 16
- learning_rate: 0.0002
- epochs: 4 (we also tried out 20 epochs)
- accelerator: 'gpu' model: GeForce RTX 3070
- Time: #######################

#### Train loss curve:

![Screenshot from 2023-09-16 12-16-01](https://github.com/CharlottePrimiceri/VP_Project/assets/115116451/c25abb75-4eb7-471f-aa21-209a60a0f5ea)

(this graph refers to 20 epochs training! See 4th epoch)

#### Model evaluation:

##### accuracy with 4 epochs: ########################## test the model
##### accuracy with 20 epochs: ########################## test the model

# Disparity Map and Depth Map

Initially we worked on Disparity Map method in order to estimate distance of detected objects, because of its reliablity and accuracy to identify the shapes. 
But the model required a pair of images for each frame (one frome the "left eye" and the other for the "right eye"), this issue made the model unsuitable for the U-Net, that is trained with a dataset composed by "monocular" images.
Beacuse of this we moved on to a more suitable method so we chose MiDaS model, that computes depth map starting from a single image.
Even if MiDaS provides a low quality map compared to the Disparity Map, it comes with better scalability that made the combination between Unet mask and distance map very easy.

![image](https://github.com/CharlottePrimiceri/VP_Project/assets/114931709/d8e56dea-1519-439b-a00e-9300640f96fe)


# Optical flow

Unfortunately, for our initial lack of gpu power, we couldn't implement the optical flow computation through a CNN such as FlowNet2. So we first implement Lucas-Kanade's Optical Flow (as shown in lucas_kanade_optical_flow.ipynb) from scratch to learn how to show its magnitude and orientation and to compute the mean of the velocity of an object as a future work. After that, we decided to estimate it through its OpenCv function (see the code in optical_flow_trajectories.py) to track the feature of a moving object in a video. 

Our Lucas-Kanade-Optical flow is limited by the moving camera mounted on the car. This movement determines that all the image is covered by red vectors (as you can see from the Colab Notebook we posted).

Lucas-Kanade: 

![image](https://github.com/CharlottePrimiceri/VP_Project/assets/114931709/18f40f4e-1320-4093-a297-e269d041c165)

![image](https://github.com/CharlottePrimiceri/VP_Project/assets/114931709/8e937b50-5b61-4ee2-ae30-7756f9fcd055)

![image](https://github.com/CharlottePrimiceri/VP_Project/assets/114931709/1b2c4fa3-682e-4213-91ad-b47a7beb1a8a)

# Pre-Processing videos

Our chosen videos are pre-processed (divided in frames) and passed both through the segmentation model of the Mask R-CNN and Pytorch U-Net. 
In order to clear the view of the rgb segmentation of the Mask R-CNN we set the focus only on the main category that appears in the videos (for the first one only pedestrian and the second one only cars).
The frames are sorted in the right order and then reunited to generate the new videos. This process is shown in MaskRCNN_and_Unet_segmented_video.ipynb file in the segmented_video folder where we can also find all the final videos.

# Compute Trajectories

We chose two kind of videos: the first one is a snippet of a longer video from the Cityscapes site, where the camera is moving torwards some people that are passing a street. Here we saw that the main problem is the fact that in this moving scenes also other objects in the background, in which we are not interested, are tracked by the algorithm because the camera it's moving! So we pick another video with a static camera and cars moving. As we can see the number of trajectories only follow them. However the presence of the bounded boxes and the category name that appear for each instance maybe cause some disturbance to the algorithm.

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


