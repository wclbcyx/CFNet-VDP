import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines
from . import flow_transforms
import torchvision
import cv2
import copy
import os 
from scipy.ndimage import gaussian_filter
import random
from torchvision import transforms

import matplotlib.pyplot as plt
feed_width = 640
feed_height = 480

process_width = feed_width
process_height = feed_height
def get_occlusion_mask(shifted):
    mask_up = shifted > 0
    mask_down = shifted > 0
    #print(mask_down)

    shifted_up = np.ceil(shifted)
    shifted_down = np.floor(shifted)
    # 如果多个点落在一个位置，选视差最大的。
    for col in range(process_width - 2):
        loc = shifted[:, col:col + 1]  # keepdims
        loc_up = np.ceil(loc)
        loc_down = np.floor(loc)

        _mask_down = ((shifted_down[:, col + 2:] != loc_down) * (
            (shifted_up[:, col + 2:] != loc_down))).min(-1)
        _mask_up = ((shifted_down[:, col + 2:] != loc_up) * (
            (shifted_up[:, col + 2:] != loc_up))).min(-1)

        mask_up[:, col] = mask_up[:, col] * _mask_up
        mask_down[:, col] = mask_down[:, col] * _mask_down

    mask = mask_up + mask_down
    return mask


def project_image(image, disp_map):
    image = np.array(image)
    # background_image = np.array(background_image)
    xs, ys = np.meshgrid(np.arange(process_width), np.arange(process_height))
    #print(xs)
    # set up for projection
    warped_image = np.zeros_like(image).astype(float)
    warped_image = np.stack([warped_image] * 2, 0)
    pix_locations = xs - disp_map
    #print(pix_locations)

    # find where occlusions are, and remove from disparity map
    mask = get_occlusion_mask(pix_locations)
    #print(mask)
    # 用来输出mask
    occ_mask = mask.copy()
    # inputs['occ_mask'] = mask
    masked_pix_locations = pix_locations * mask - process_width * (1 - mask)

    # do projection - linear interpolate up to 1 pixel away
    weights = np.ones((2, process_height, process_width)) * 10000

    for col in range(process_width - 1, -1, -1):
        loc = masked_pix_locations[:, col]
        loc_up = np.ceil(loc).astype(int)
        loc_down = np.floor(loc).astype(int)
        weight_up = loc_up - loc
        weight_down = 1 - weight_up

        mask = loc_up >= 0
        mask[mask] = \
            weights[0, np.arange(process_height)[mask], loc_up[mask]] > weight_up[mask]
        weights[0, np.arange(process_height)[mask], loc_up[mask]] = \
            weight_up[mask]
        warped_image[0, np.arange(process_height)[mask], loc_up[mask]] = \
            image[:, col][mask] / 255.

        mask = loc_down >= 0
        mask[mask] = \
            weights[1, np.arange(process_height)[mask], loc_down[mask]] > weight_down[mask]
        weights[1, np.arange(process_height)[mask], loc_down[mask]] = weight_down[mask]
        warped_image[1, np.arange(process_height)[mask], loc_down[mask]] = \
            image[:, col][mask] / 255.

    weights /= weights.sum(0, keepdims=True) + 1e-7  # normalise
    weights = np.expand_dims(weights, -1)
    warped_image = warped_image[0] * weights[1] + warped_image[1] * weights[0]
    warped_image *= 255.

    # now fill occluded regions with random background
    # if background_image is not None:
    #     warped_image[warped_image.max(-1) == 0] = background_image[warped_image.max(-1) == 0]
    warped_image = warped_image.astype(np.uint8)

    return warped_image, occ_mask



class KITTIDataset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 65536.
        return data
    def eraser_transform(self, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img2.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img2
    def stereo_aug_sym(self, img1, img2, disp):
        
        
        #img1,img2,disp = self.spatial_transform(img1, img2, disp)
        
        wW,hH = img1.size
        
        img = Image.new('RGB', (2*wW, hH), (255,0,0))
        img.paste(img1,(0,0)) 
        img.paste(img2,(wW,0)) 
        
        w,h = img.size
        
        
        stereo_brightness = (0.5, 1.5)
        stereo_contrast = (0.5, 1.5)
        stereo_saturation = (0.5, 1.5)
        stereo_hue = (-0.05, 0.05)
        
        
        # random_number_color = random.random()
        random_number = random.random()
        if random_number<=0.8:
            aug_color = transforms.ColorJitter(stereo_brightness, stereo_contrast, stereo_saturation,stereo_hue)
            img = aug_color(img)
        
        img_np = np.array(img)
        
        random_number = random.random()    
        if random_number<=0.8:
            noise = np.random.randn(h, w, 3) / 50.0  
            #print(img_np.shape)
            img_np = np.clip(img_np / 255 + noise, 0, 1) * 255
        
        random_number = random.random()
        if random_number<=0.8:
        
            img_np = gaussian_filter(img_np,sigma=random.random())
       

        img = Image.fromarray(img_np.astype('uint8'))
        
        trans_img1 = img.crop((0,0,wW,hH))
        trans_img2 = img.crop((wW,0,2*wW,hH))
        return trans_img1, trans_img2, disp
    
    def stereo_aug_asym(self, img):
        w,h = img.size
        
        stereo_brightness = (0.5, 1.5)
        stereo_contrast = (0.5, 1.5)
        stereo_saturation = (0.5, 1.5)
        stereo_hue = (-0.05, 0.05)
        aug_color = transforms.ColorJitter(stereo_brightness, stereo_contrast, stereo_saturation,stereo_hue)
        
        # random_number_color = random.random()
        random_number = random.random()
        if random_number<=0.8:
            img = aug_color(img)
        
        img_np = np.array(img)
        # random_number = random.random()  
        # if random_number<=0.8:
        #     img_np = self.eraser_transform(img_np)
           
        random_number = random.random()
        if random_number<=0.8:    
            noise = np.random.randn(h, w, 3) / 50.0
            
            img_np = np.clip(img_np / 255 + noise, 0, 1) * 255
        
        random_number = random.random()
        if random_number<=0.8:
            img_np = gaussian_filter(img_np,sigma=random.random())
        img = Image.fromarray(img_np.astype('uint8'))
        
        return img
    # def RGB2GRAY(self, img):
    #     imgG = copy.deepcopy(img)
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     imgG[:, :, 0] = img
    #     imgG[:, :, 1] = img
    #     imgG[:, :, 2] = img
    #     return imgG

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        #right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        # left_img = self.RGB2GRAY(left_img)
        # right_img = self.RGB2GRAY(right_img)
        #right_img = right_img.copy()


        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
        else:
            disparity = None
        left_img = left_img.copy()

        # 生成 185 到 215 之间的随机数
        random_multiplier = random.randint(185, 215)
        disparity = disparity * random_multiplier
        # disparity = disparity * 192 
        # left_img = np.array(left_img)
        warp_image, occ_mask = project_image(left_img, disparity)
        right_img = Image.fromarray(warp_image.astype('uint8'))  # numpy -> PIL


        # output_dir = "output_images"
        # os.makedirs(output_dir, exist_ok=True)
        
        # # 保存 left_img

        # plt.imsave(os.path.join(output_dir, f"left_img1.png"), left_img)
        
        # # 保存 disparity (视差图通常用颜色映射增强可视化)
        # plt.imsave(os.path.join(output_dir, f"disparity.png"), disparity, cmap='jet')
        
        # # 保存 warp_image
        # plt.imsave(os.path.join(output_dir, f"warp_image.png"), warp_image)
        
        if self.training:
            th, tw = 256, 512
            #th, tw = 320, 704
            left_img, right_img,disparity = self.stereo_aug_sym(left_img, right_img,disparity)
            right_img = self.stereo_aug_asym(right_img)
            # random_brightness = np.random.uniform(0.5, 2.0, 2)
            # random_gamma = np.random.uniform(0.8, 1.2, 2)
            # random_contrast = np.random.uniform(0.8, 1.2, 2)
            # left_img = torchvision.transforms.functional.adjust_brightness(left_img, random_brightness[0])
            # left_img = torchvision.transforms.functional.adjust_gamma(left_img, random_gamma[0])
            # left_img = torchvision.transforms.functional.adjust_contrast(left_img, random_contrast[0])
            # right_img = torchvision.transforms.functional.adjust_brightness(right_img, random_brightness[1])
            # right_img = torchvision.transforms.functional.adjust_gamma(right_img, random_gamma[1])
            # right_img = torchvision.transforms.functional.adjust_contrast(right_img, random_contrast[1])
            right_img = np.asarray(right_img)
            left_img = np.asarray(left_img)

            # w, h  = left_img.size
            # th, tw = 256, 512
            #
            # x1 = random.randint(0, w - tw)
            # y1 = random.randint(0, h - th)
            #
            # left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            # right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
            # dataL = dataL[y1:y1 + th, x1:x1 + tw]
            # right_img = np.asarray(right_img)
            # left_img = np.asarray(left_img)

            # geometric unsymmetric-augmentation
            # angle = 0;
            # px = 0
            # if np.random.binomial(1, 0.5):
            #     # angle = 0.1;
            #     # px = 2
            #     angle = 0.05
            #     px = 1
            co_transform = flow_transforms.Compose([
                # flow_transforms.RandomVdisp(angle, px),
                # flow_transforms.Scale(np.random.uniform(self.rand_scale[0], self.rand_scale[1]), order=self.order),
                flow_transforms.RandomCrop((th, tw)),
            ])
            augmented, disparity = co_transform([left_img, right_img], disparity)
            left_img = augmented[0]
            right_img = augmented[1]
            right_img = right_img.copy()
            #right_img.flags.writeable = True
            if np.random.binomial(1,0.2):
              sx = int(np.random.uniform(35,100))
              sy = int(np.random.uniform(25,75))
              cx = int(np.random.uniform(sx,right_img.shape[0]-sx))
              cy = int(np.random.uniform(sy,right_img.shape[1]-sy))
              right_img[cx-sx:cx+sx,cy-sy:cy+sy] = np.mean(np.mean(right_img,0),0)[np.newaxis,np.newaxis]

            # to tensor, normalize
            disparity = np.ascontiguousarray(disparity, dtype=np.float32)
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity}
        else:
            w, h = left_img.size

            # normalize
            processed = get_transform()
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()

            # pad to size 640x480
            top_pad = 480 - h
            right_pad = 640 - w
            assert top_pad >= 0 and right_pad >= 0
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            # pad disparity gt
            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            if disparity is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]
                        }
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}
