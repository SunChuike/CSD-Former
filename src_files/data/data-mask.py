from cProfile import label
import os
import random
import torch
from randaugment import RandAugment
import torchvision.transforms as transforms
from PIL import ImageDraw, Image
from src_files.data.handlers import weather_handler, VOC2007_handler, COCO2014_handler, VG256_handler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2

np.set_printoptions(suppress=True)

HANDLER_DICT = {
    'Multi-label-dataset': weather_handler,
    'voc2007': VOC2007_handler,
    'coco2014': COCO2014_handler,
    'vg256': VG256_handler
}

def get_datasets(args, patch=False):

    if patch:
        train_transform = TransformPatch_Train(args)
        val_transform = TransformPatch_Val(args)

    else:
        train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        CutoutPIL(cutout_factor=0.5),
        RandAugment(),
        transforms.ToTensor()])

        val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()])

    # load data:
    source_data = load_data(args.data_dir)
	
    data_handler = HANDLER_DICT[args.data_name]

    train_dataset = data_handler(source_data['train']['images'], source_data['train']['labels'], args.data_dir, transform=train_transform)
    
    val_dataset = data_handler(source_data['val']['images'], source_data['val']['labels'], args.data_dir, transform=val_transform)

    return train_dataset, val_dataset



def load_data(base_path):
    data = {}
    for phase in ['train', 'val']:
        data[phase] = {}
        data[phase]['labels'] = np.load(os.path.join(base_path, 'formatted_{}_labels.npy'.format(phase)))
        data[phase]['images'] = np.load(os.path.join(base_path, 'formatted_{}_images.npy'.format(phase)))
    return data

class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


class TransformPatch_Train(object):
    def __init__(self, args):
        self.n_grid = args.n_grid
        self.image_size = args.image_size
        self.padding_size = int(self.image_size * 1.3)  # 调整 padding
        self.mask_ratio = 0.20
        self.augmentations = A.Compose([
            A.PadIfNeeded(min_height=self.padding_size, min_width=self.padding_size, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.2, p=0.7),
            A.GaussianBlur(blur_limit=(3, 7), p=0.8),
            A.HorizontalFlip(p=0.5),
            A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=0.3),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5, p=0.3),
            A.CenterCrop(self.image_size, self.image_size),
        ])
        self.to_tensor = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def random_mask(self, img):
        
        img_w, img_h = img.size
        mask_w = int(img_w * np.sqrt(self.mask_ratio))
        mask_h = int(img_h * np.sqrt(self.mask_ratio))
        
        mask_x = random.randint(0, img_w - mask_w)
        mask_y = random.randint(0, img_h - mask_h)
        
        mask_img = Image.new('RGB', (img_w, img_h), color=(0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_img)
        mask_draw.rectangle((mask_x, mask_y, mask_x + mask_w, mask_y + mask_h), fill=(0, 0, 0))

        masked_img = Image.composite(mask_img, img, mask_img.convert('L'))
        return masked_img

    def __call__(self, img):
        img = img.resize((self.image_size, self.image_size))
        strong_list = []
        img = self.random_mask(img)
        # 整图增强
        img_np = np.array(img)
        augmented = self.augmentations(image=img_np)
        image = augmented['image']
        image = self.to_tensor(image=image)['image']
        strong_list.append(image)

        # patches 不增强
        x_order = np.random.permutation(self.n_grid)
        y_order = np.random.permutation(self.n_grid)
        grid_size_x = img.size[0] // self.n_grid
        grid_size_y = img.size[1] // self.n_grid
        for i in x_order:
            for j in y_order:
                x_offset = i * grid_size_x
                y_offset = j * grid_size_y
                patch = img.crop((x_offset, y_offset, x_offset + grid_size_x, y_offset + grid_size_y))
                patch = patch.resize((self.image_size, self.image_size))
                patch = self.to_tensor(image=np.array(patch))['image']
                strong_list.append(patch)

        return strong_list

class TransformPatch_Val(object):
    def __init__(self, args):
        self.n_grid = args.n_grid
        self.image_size = args.image_size
        
        self.weak = transforms.Compose([
                        transforms.Resize((args.image_size, args.image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        # normalize, # no need, toTensor does normalization
                    ])

    def __call__(self, img):
        weak_list = [self.weak(img)]

        # Append patches
        img = img.resize((self.image_size, self.image_size))

        # To permute the order for local patched
        x_order = np.random.permutation(self.n_grid)
        y_order = np.random.permutation(self.n_grid)

        grid_size_x = img.size[0] // self.n_grid
        grid_size_y = img.size[1] // self.n_grid
        
        for i in x_order:
            for j in y_order:
                x_offset = i * grid_size_x
                y_offset = j * grid_size_y
                patch = img.crop((x_offset, y_offset, x_offset + grid_size_x, y_offset + grid_size_y))
                # Append patches
                weak_list.append(self.weak(patch))
        
        return weak_list




