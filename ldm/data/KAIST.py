import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import torchvision.transforms.functional as tf
from copy import deepcopy

def random_crop(image1, image2):
    min_ratio = 0.5
    max_ratio = 1

    w, h = image1.size
    ratio = random.random()
    scale = min_ratio + ratio * (max_ratio - min_ratio)
    new_h = int(h * scale)
    new_w = int(w * scale)
    y = np.random.randint(0, h - new_h)
    x = np.random.randint(0, w - new_w)

    image1 = image1.crop((x, y, x + new_w, y + new_h))
    image2 = image2.crop((x, y, x + new_w, y + new_h))

    return image1, image2


class my_transform_crop():
    def __init__(self, crop_p=0.5):
        self.crop_p=crop_p
    def crop_enhance(self, image1, image2):
        if random.random() <= self.crop_p:
            image1, image2 = random_crop(image1, image2)
        return image1, image2

class my_transform_flip():
    def __init__(self, flip_p=0.5):
        self.flip_p = flip_p
        
    def flip_enhance(self, image1, image2):
        if random.random() <= self.flip_p:
            image1 = np.flip(image1, axis=1)
            image2 = np.flip(image2, axis=1)
        return image1, image2

class my_transform_gray():
    def __init__(self, gray_p=0.5):
        self.gray_p = gray_p
        
    def gray_enhance(self, image1):
        if random.random() <= self.gray_p:
            gray = deepcopy(0.299*image1[:,:,0] + 0.587*image1[:,:,1] + 0.114*image1[:,:,2])
            image1[:,:,0] = gray
            image1[:,:,1] = gray
            image1[:,:,2] = gray
        return image1

class KAISTBase(Dataset):
    def __init__(self,
                 ir_dir,       # 红外图片文件夹，例如 /data/KAIST/train/lwir
                 vi_dir,       # 可见光图片文件夹，例如 /data/KAIST/train/visible
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5):

        EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

        ir_files = sorted([
            f for f in os.listdir(ir_dir)
            if os.path.splitext(f)[1].lower() in EXTS
        ])

        self._length = len(ir_files)
        self.labels = {
            "ir_file_path_": [os.path.join(ir_dir, f) for f in ir_files],
            "vi_file_path_": [os.path.join(vi_dir, f) for f in ir_files],
        }

        self.size = size
        self.interpolation = {"bicubic": PIL.Image.BICUBIC,
                              "bilinear": PIL.Image.BILINEAR,
                              "lanczos": PIL.Image.LANCZOS}[interpolation]
        self.flip_enhance  = my_transform_flip(flip_p=flip_p)
        self.crop_enhance  = my_transform_crop(crop_p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {k: self.labels[k][i] for k in self.labels}

        def load_rgb(path):
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.array(img).astype(np.uint8)
            crop = min(arr.shape[0], arr.shape[1])
            h, w = arr.shape[0], arr.shape[1]
            arr = arr[(h-crop)//2:(h+crop)//2, (w-crop)//2:(w+crop)//2]
            return Image.fromarray(arr)

        image_ir = load_rgb(example["ir_file_path_"])
        image_vi = load_rgb(example["vi_file_path_"])

        if self.size is not None:
            image_ir, image_vi = self.crop_enhance.crop_enhance(image_ir, image_vi)
            image_ir = image_ir.resize((self.size, self.size), resample=self.interpolation)
            image_vi = image_vi.resize((self.size, self.size), resample=self.interpolation)

        image_ir = np.array(image_ir).astype(np.uint8)
        image_vi = np.array(image_vi).astype(np.uint8)
        image_ir, image_vi = self.flip_enhance.flip_enhance(image_ir, image_vi)

        return {
            "image":       (image_ir / 127.5 - 1.0).astype(np.float64),
            "conditional": (image_vi / 127.5 - 1.0).astype(np.float64),
        }


class KAISTTrain(KAISTBase):
    def __init__(self, **kwargs):
        super().__init__(
            ir_dir="/your/path/KAIST/train/lwir",
            vi_dir="/your/path/KAIST/train/visible",
            **kwargs
        )

class KAISTVal(KAISTBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(
            ir_dir="/your/path/KAIST/test/lwir",
            vi_dir="/your/path/KAIST/test/visible",
            flip_p=flip_p,
            **kwargs
        )
