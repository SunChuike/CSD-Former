from cProfile import label
import os
import random
import torch
from randaugment import RandAugment
import torchvision.transforms as transforms
from PIL import ImageDraw, Image
# 假设这些 handler 存在且已正确导入
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
        train_transform = TransformPatchTrain(args)
        val_transform = TransformPatchVal(args)

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


# # --- 新的掩码生成器 ---
class RandomAreaMaskerPIL:
    
    """
    在 PIL Image 上应用随机小块掩码，直到达到指定的总掩码面积比例。
    """
    def __init__(self, total_area_ratio=0.20, min_mask_scale=0.005, max_mask_scale=0.04, aspect_ratio_range=(0.5, 2.0)):
        self.total_area_ratio = total_area_ratio
        self.min_mask_scale = min_mask_scale
        self.max_mask_scale = max_mask_scale
        self.aspect_ratio_range = aspect_ratio_range
        self.fill_color = (0, 0, 0)

    def __call__(self, img):
        img_w, img_h = img.size
        img_draw = ImageDraw.Draw(img)
        target_mask_area = img_w * img_h * self.total_area_ratio
        current_masked_area = 0
        max_attempts = 50 
        for _ in range(max_attempts):
            if current_masked_area >= target_mask_area:
                break
            mask_area = random.uniform(self.min_mask_scale, self.max_mask_scale) * img_w * img_h
            aspect_ratio = random.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1])
            mask_w = int(np.sqrt(mask_area * aspect_ratio))
            mask_h = int(np.sqrt(mask_area / aspect_ratio))
            if mask_w < 1 or mask_h < 1:
                continue
            x1 = random.randint(0, img_w - mask_w)
            y1 = random.randint(0, img_h - mask_h)
            img_draw.rectangle([x1, y1, x1 + mask_w, y1 + mask_h], fill=self.fill_color)
            current_masked_area += mask_w * mask_h
        return img

# --- 优化的数据变换类 (已修改) ---
class TransformPatchTrain(object):
    """
    为 Swin Transformer 训练设计的变换流程 (Global View + Local Patches)
    流程: 掩码 -> 强数据增强 -> 提取无增强的Patches
    """
    def __init__(self, args):
        self.n_grid = args.n_grid
        self.image_size = args.image_size

        self.masker = RandomAreaMaskerPIL(
            total_area_ratio=0.35,
            min_mask_scale=0.005,
            max_mask_scale=0.05,
            aspect_ratio_range=(0.3, 3.3)
        )

        self.augmentations = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.HorizontalFlip(p=0.5),
            A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=0.2),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, p=0.2),
        ])

        self.to_tensor_and_normalize = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        # --- 全局视图 (Global View) 处理 ---
        augmented_img_pil = img.copy()
        augmented_img_pil = self.masker(augmented_img_pil)
        augmented_img_np = np.array(augmented_img_pil)
        augmented_img_np = self.augmentations(image=augmented_img_np)['image']
        global_view_tensor = self.to_tensor_and_normalize(image=augmented_img_np)['image']
        
        transformed_outputs = [global_view_tensor]

        # --- 局部视图 (Local Patches) 处理 ---
        grid_size = self.image_size // self.n_grid
        
        for i in range(self.n_grid):
            for j in range(self.n_grid):
                x_offset = i * grid_size
                y_offset = j * grid_size
                patch_pil = img.crop((x_offset, y_offset, x_offset + grid_size, y_offset + grid_size))
                
                # ====================  关键修改处  ====================
                # 将裁剪出的小 patch 重新 resize 到与全局视图相同的尺寸
                # 这是为了与第一个代码块的输出维度保持一致
                patch_pil = patch_pil.resize((self.image_size, self.image_size), Image.BILINEAR)
                patch_pil = self.masker(patch_pil)
                # ======================================================

                patch_np = np.array(patch_pil)
                # patch_np = self.augmentations(image=patch_np)['image']
                patch_tensor = self.to_tensor_and_normalize(image=patch_np)['image']
                transformed_outputs.append(patch_tensor)
        
        return transformed_outputs

class TransformPatchVal(object):
    """
    验证集的变换流程，仅做 Resize 和 Normalize，保持确定性。
    (注意：为了和第一个代码块的 Val 部分保持一致，这里的 Patches 也需要 resize)
    """
    def __init__(self, args):
        self.n_grid = args.n_grid
        self.image_size = args.image_size
        self.weak_transform = A.Compose([
            # 这个 resize 将作用于整图和 resize 后的 patch
            A.Resize(height=self.image_size, width=self.image_size), 
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        # 全局视图
        img_np = np.array(img)
        # 直接对原图应用变换，它会被 resize 到 image_size
        global_view_tensor = self.weak_transform(image=img_np)['image']
        transformed_outputs = [global_view_tensor]

        # 局部视图 (Patches)
        # 先将原图 resize，再从中裁剪
        img_resized_pil = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        grid_size = self.image_size // self.n_grid
        
        for i in range(self.n_grid):
            for j in range(self.n_grid):
                x_offset = i * grid_size
                y_offset = j * grid_size
                patch_pil = img_resized_pil.crop((x_offset, y_offset, x_offset + grid_size, y_offset + grid_size))
                patch_np = np.array(patch_pil)
                # 对 patch 应用变换，它也会被 resize 到 image_size
                patch_tensor = self.weak_transform(image=patch_np)['image']
                transformed_outputs.append(patch_tensor)
        
        return transformed_outputs