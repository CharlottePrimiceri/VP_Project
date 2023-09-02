# Vision and Perception - Final Project

For our final project we have focused on the OBJECT DETECTION TASK performed on the Cityscapes Dataset.

Our aim was to investigate wheter the segmentation task could be improved or not training models on the depth maps and the optical flow graphic representations in addiction to original images.

We have performed image segmentation on a video taken from Cityscapes Dataset using three different models:
  1)  Mask-RCNN pre-trained on COCO Dataset (Detectron2 implementation)
  2)  Pytorch Unet with double input channel (that we trained from start)
  3)  Pytorch Unet with single input channel (as original, that we trained from start)

References:
- Cityscapes dataset: https://www.cityscapes-dataset.com
- Mask-RCNN: https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf
- Mask-RCNN with disparity map and optical flow: https://github.com/thiagortk/maskRCNN-DisparityMap-OpicalFlow
- Detecron2: https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md


