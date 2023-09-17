import os.path
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as TF
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.init as init
import torchvision.utils as v_utils
from torch.autograd import Variable


torch.backends.cudnn.benchmark = True


#################################################################
############# MODEL BLOCKS ######################################
#################################################################

def conv_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def conv_trans_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_2(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model


def conv_block_3(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        conv_block(out_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model



#################################################################
################   UNET 1   #####################################
#################################################################

class unet_without_last_layers_1(nn.Module):
        
        def __init__(self, in_dim, out_dim, num_filter):
                super(unet_without_last_layers_1, self).__init__()
                self.in_dim = in_dim
                self.out_dim = out_dim
                self.num_filter = num_filter
                act_fn = nn.LeakyReLU(0.2, inplace=True)

                self.down_1 = conv_block_2(self.in_dim, self.num_filter, act_fn)
                self.pool_1 = maxpool()
                self.down_2 = conv_block_2(self.num_filter*1, self.num_filter*2, act_fn)
                self.pool_2 = maxpool()
                self.down_3 = conv_block_2(self.num_filter*2, self.num_filter*4, act_fn)
                self.pool_3 = maxpool()
                self.down_4 = conv_block_2(self.num_filter*4, self.num_filter*8, act_fn)
                self.pool_4 = maxpool()

                self.bridge = conv_block_2(self.num_filter*8, self.num_filter*16, act_fn)

                self.trans_1 = conv_trans_block(self.num_filter*16, self.num_filter*8, act_fn)
                self.up_1 = conv_block_2(self.num_filter*16, self.num_filter*8, act_fn)
                self.trans_2 = conv_trans_block(self.num_filter*8, self.num_filter*4, act_fn)
                self.up_2 = conv_block_2(self.num_filter*8, self.num_filter*4, act_fn)
                self.trans_3 = conv_trans_block(self.num_filter*4, self.num_filter*2, act_fn)
                self.up_3 = conv_block_2(self.num_filter*4, self.num_filter*2, act_fn)
                self.trans_4 = conv_trans_block(self.num_filter*2, self.num_filter*1, act_fn)
                self.up_4 = conv_block_2(self.num_filter*2, self.num_filter*1, act_fn)


        def forward(self, input):
                down_1 = self.down_1(input)
                pool_1 = self.pool_1(down_1)
                down_2 = self.down_2(pool_1)
                pool_2 = self.pool_2(down_2)
                down_3 = self.down_3(pool_2)
                pool_3 = self.pool_3(down_3)
                down_4 = self.down_4(pool_3)
                pool_4 = self.pool_4(down_4)

                bridge = self.bridge(pool_4)

                trans_1 = self.trans_1(bridge)
                concat_1 = torch.cat([trans_1, down_4], dim=1)
                up_1 = self.up_1(concat_1)
                trans_2 = self.trans_2(up_1)
                concat_2 = torch.cat([trans_2, down_3], dim=1)
                up_2 = self.up_2(concat_2)
                trans_3 = self.trans_3(up_2)
                concat_3 = torch.cat([trans_3, down_2], dim=1)
                up_3 = self.up_3(concat_3)
                trans_4 = self.trans_4(up_3)
                concat_4 = torch.cat([trans_4, down_1], dim=1)
                up_4 = self.up_4(concat_4)
                
                return up_4


#################################################################
################   UNET 2   #####################################
#################################################################

class unet_without_last_layers_2(nn.Module):
        
        def __init__(self, in_dim, out_dim, num_filter):
                super(unet_without_last_layers_2, self).__init__()
                self.in_dim = in_dim
                self.out_dim = out_dim
                self.num_filter = num_filter
                act_fn = nn.LeakyReLU(0.2, inplace=True)

                self.down_1 = conv_block_2(self.in_dim, self.num_filter, act_fn)
                self.pool_1 = maxpool()
                self.down_2 = conv_block_2(self.num_filter*1, self.num_filter*2, act_fn)
                self.pool_2 = maxpool()
                self.down_3 = conv_block_2(self.num_filter*2, self.num_filter*4, act_fn)
                self.pool_3 = maxpool()
                self.down_4 = conv_block_2(self.num_filter*4, self.num_filter*8, act_fn)
                self.pool_4 = maxpool()

                self.bridge = conv_block_2(self.num_filter*8, self.num_filter*16, act_fn)

                self.trans_1 = conv_trans_block(self.num_filter*16, self.num_filter*8, act_fn)
                self.up_1 = conv_block_2(self.num_filter*16, self.num_filter*8, act_fn)
                self.trans_2 = conv_trans_block(self.num_filter*8, self.num_filter*4, act_fn)
                self.up_2 = conv_block_2(self.num_filter*8, self.num_filter*4, act_fn)
                self.trans_3 = conv_trans_block(self.num_filter*4, self.num_filter*2, act_fn)
                self.up_3 = conv_block_2(self.num_filter*4, self.num_filter*2, act_fn)
                self.trans_4 = conv_trans_block(self.num_filter*2, self.num_filter*1, act_fn)
                self.up_4 = conv_block_2(self.num_filter*2, self.num_filter*1, act_fn)


        def forward(self, input):
                down_1 = self.down_1(input)
                pool_1 = self.pool_1(down_1)
                down_2 = self.down_2(pool_1)
                pool_2 = self.pool_2(down_2)
                down_3 = self.down_3(pool_2)
                pool_3 = self.pool_3(down_3)
                down_4 = self.down_4(pool_3)
                pool_4 = self.pool_4(down_4)

                bridge = self.bridge(pool_4)

                trans_1 = self.trans_1(bridge)
                concat_1 = torch.cat([trans_1, down_4], dim=1)
                up_1 = self.up_1(concat_1)
                trans_2 = self.trans_2(up_1)
                concat_2 = torch.cat([trans_2, down_3], dim=1)
                up_2 = self.up_2(concat_2)
                trans_3 = self.trans_3(up_2)
                concat_3 = torch.cat([trans_3, down_2], dim=1)
                up_3 = self.up_3(concat_3)
                trans_4 = self.trans_4(up_3)
                concat_4 = torch.cat([trans_4, down_1], dim=1)
                up_4 = self.up_4(concat_4)
                
                return up_4

#################### MODEL #######################################
        
class UnetGenerator(nn.Module):

    def __init__(self, in_dim, out_dim, num_filter):
        super(UnetGenerator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        self.unet_first_part_1 = unet_without_last_layers_1(self.in_dim, self.out_dim, self.num_filter)
        self.unet_first_part_2 = unet_without_last_layers_2(self.in_dim, self.out_dim, self.num_filter)
        self.layer_out = nn.Sequential(
            nn.Conv2d(128, self.out_dim, 3, 1, 1),
            nn.Tanh(),)

    def forward(self, input_image, input_depth):

        up_4_image = self.unet_first_part_1(input_image)
        up_4_depth = self.unet_first_part_2(input_depth)
        up_4_merged = torch.cat((up_4_image, up_4_depth), 1)
        out = self.layer_out(up_4_merged)

        return out

##########################################################################################################
                              # DATAMODULE #
##########################################################################################################

class CityscapesDataset(Dataset):

    def __init__(self, root, split='train', mode='fine'):

        self.root = os.path.expanduser(root)
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.depth_images_dir = os.path.join(self.root, 'depth_images', split)  ##ADDED PATH TO DEPTH_IMAGES
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.split = split
        self.images = []
        self.depth_images = []                                                  ##ADDED LIST OF DEPTH_IMAGES
        self.targets = []  
        self.mapping = {
            0: 0,  # unlabeled
            1: 0,  # ego vehicle
            2: 0,  # rect border
            3: 0,  # out of roi
            4: 0,  # static
            5: 0,  # dynamic
            6: 0,  # ground
            7: 1,  # road
            8: 0,  # sidewalk
            9: 0,  # parking
            10: 0,  # rail track
            11: 0,  # building
            12: 0,  # wall
            13: 0,  # fence
            14: 0,  # guard rail
            15: 0,  # bridge
            16: 0,  # tunnel
            17: 0,  # pole
            18: 0,  # polegroup
            19: 0,  # traffic light
            20: 0,  # traffic sign
            21: 0,  # vegetation
            22: 0,  # terrain
            23: 2,  # sky
            24: 0,  # person
            25: 0,  # rider
            26: 3,  # car
            27: 0,  # truck
            28: 0,  # bus
            29: 0,  # caravan
            30: 0,  # trailer
            31: 0,  # train
            32: 0,  # motorcycle
            33: 0,  # bicycle
            -1: 0  # licenseplate
        }
        self.mappingrgb = {
            0: (255, 0, 0),  # unlabeled
            1: (255, 0, 0),  # ego vehicle
            2: (255, 0, 0),  # rect border
            3: (255, 0, 0),  # out of roi
            4: (255, 0, 0),  # static
            5: (255, 0, 0),  # dynamic
            6: (255, 0, 0),  # ground
            7: (0, 255, 0),  # road
            8: (255, 0, 0),  # sidewalk
            9: (255, 0, 0),  # parking
            10: (255, 0, 0),  # rail track
            11: (255, 0, 0),  # building
            12: (255, 0, 0),  # wall
            13: (255, 0, 0),  # fence
            14: (255, 0, 0),  # guard rail
            15: (255, 0, 0),  # bridge
            16: (255, 0, 0),  # tunnel
            17: (255, 0, 0),  # pole
            18: (255, 0, 0),  # polegroup
            19: (255, 0, 0),  # traffic light
            20: (255, 0, 0),  # traffic sign
            21: (255, 0, 0),  # vegetation
            22: (255, 0, 0),  # terrain
            23: (0, 0, 255),  # sky
            24: (255, 0, 0),  # person
            25: (255, 0, 0),  # rider
            26: (255, 255, 0),  # car
            27: (255, 0, 0),  # truck
            28: (255, 0, 0),  # bus
            29: (255, 0, 0),  # caravan
            30: (255, 0, 0),  # trailer
            31: (255, 0, 0),  # train
            32: (255, 0, 0),  # motorcycle
            33: (255, 0, 0),  # bicycle
            -1: (255, 0, 0)  # licenseplate
        }

        # For example 4 classes, means we should map to the ids=(0,1,2,3)
        # This is used to specify how many outputs the network should product...
        self.num_classes = 4

    ######## MATCHING SOURCE AND LABELS GTFINE OF CITYSCAPES LEFTIMG8BIT(SOURCE X1) ########
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            depth_dir = os.path.join(self.depth_images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                self.depth_images.append(os.path.join(depth_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0], '{}_labelIds.png'.format(self.mode))
                self.targets.append(os.path.join(target_dir, target_name))
              

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of images: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Mode: {}\n'.format(self.mode)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str

    def __len__(self):
        return len(self.images)

    def mask_to_class(self, mask):
        '''
        Given the cityscapes dataset, this maps to a 0..classes numbers.
        This is because we are using a subset of all masks, so we have this "mapping" function.
        This mapping function is used to map all the standard ids into the smaller subset.
        '''
        maskimg = torch.zeros((mask.size()[0], mask.size()[1], mask.size()[2]), dtype=torch.uint8)
        for k in self.mapping:
            maskimg[mask == k] = self.mapping[k]
        return maskimg

    def mask_to_rgb(self, mask):
        '''
        Given the Cityscapes mask file, this converts the ids into rgb colors.
        This is needed as we are interested in a sub-set of labels, thus can't just use the
        standard color output provided by the dataset.
        '''
        rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1], mask.size()[2]), dtype=torch.uint8)
        for k in self.mappingrgb:
            rgbimg[0][mask == k] = self.mappingrgb[k][0]
            rgbimg[1][mask == k] = self.mappingrgb[k][1]
            rgbimg[2][mask == k] = self.mappingrgb[k][2]
        return rgbimg

    def class_to_rgb(self, mask):
        '''
        This function maps the classification index ids into the rgb.
        For example after the argmax from the network, you want to find what class
        a given pixel belongs too. This does that but just changes the color
        so that we can compare it directly to the rgb groundtruth label.
        '''
        mask2class = dict((v, k) for k, v in self.mapping.items()) 
        rgbimg = torch.zeros((3, mask.size()[1], mask.size()[2]), dtype=torch.uint8)
        for k in mask2class:
            rgbimg[0][mask == k] = self.mappingrgb[mask2class[k]][0]
            rgbimg[1][mask == k] = self.mappingrgb[mask2class[k]][1]
            rgbimg[2][mask == k] = self.mappingrgb[mask2class[k]][2]
        return rgbimg

    def __getitem__(self, index):

        # first load the RGB image
        image = Image.open(self.images[index]).convert('RGB')

        # load depth image
        depth_image = Image.open(self.depth_images[index]).convert('RGB')   

        # next load the target
        target = Image.open(self.targets[index]).convert('L')

        # Data Augmentation
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(image,output_size=(128, 256))
        transforms = transforms.Compose([torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Resize((128, 256), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),])
      
        transforms_targs = transforms.Compose([torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Resize((128, 256), interpolation=torchvision.transforms.InterpolationMode.NEAREST),])
        
        image = transforms(image)
        depth_image = transforms(depth_image)
        both = torch.stack((image, depth_image), dim=0)
        target = transforms_targs(target)
        target = torch.from_numpy(np.array(target, dtype=np.uint8))
        
        # convert the labels into a mask
        targetrgb = self.mask_to_rgb(target)
        targetmask = self.mask_to_class(target)
        targetmask = targetmask.long()
        targetrgb = targetrgb.long()
        
        imagergb = both[0]
        depth = both[1]
        
        return imagergb, depth, targetmask, targetrgb                                               ##ADDED RGB DEPTH_IMAGE AS FOURTH ARGUMENT

#####################################################################################################################################################

def main():
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # our CLI parser
        parser = argparse.ArgumentParser()
        parser.add_argument("--datadir", type=str, default="c:/Users/zitoc/EUREKA_GPU/cityscapes", help="directory of images")
        parser.add_argument("--batch_size", type=int, default=1, help="batch size")
        parser.add_argument("--num_gpu", type=int, default=0, help="number of gpus")
        args = parser.parse_args()

        # load the images in the folder
        dataset = CityscapesDataset(args.datadir, split='val', mode='fine')
        img_batch = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        print(dataset)

        # load pretrained model if it is there
        print("loading unet model...")
        file_model = './unet.pkl'
        if os.path.isfile(file_model):
            generator = torch.load(file_model)
            print("    - model restored from file....")
            print("    - filename = %s" % file_model)
        else:
            print("unable to load unet.pkl model file")
            exit()


        # make the result directory
        if not os.path.exists('./predict/'):
            os.makedirs('./predict/')

        # Loop through the dataset and evaluate how well the network predicts
        print("\nevaluating network (will take a while)...")

        for idx_batch, (imagergb, depth, labelmask, labelrgb), in enumerate(img_batch):

            # send to the GPU and do a forward pass
            x1 = Variable(imagergb).to(device)
            x2 = Variable(depth).to(device)
            y = generator.forward(x1, x2)

            pred_class = torch.zeros((y.size()[0], y.size()[2], y.size()[3]))
            for idx in range(0, y.size()[0]):
                pred_class[idx] = torch.argmax(y[idx], dim=0).cpu().int()

            pred_class = pred_class.unsqueeze(1).float()

            # debug saving generated classes to file
            v_utils.save_image(pred_class.float()/y.size()[1], "./predict/gen_image_{}_{}.png".format(0, idx_batch))
            v_utils.save_image(x1.cpu().data, "./predict/original_image_{}_{}.png".format(0, idx_batch))
            v_utils.save_image(x2.cpu().data, "./predict/depth_{}_{}.png".format(0, idx_batch))

if __name__=='__main__':
     main()