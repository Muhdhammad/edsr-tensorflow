import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import Sequence
from .augmentations import crop_patches, flip_images, rotate_images

def create_data_splits(image_dir, batch_size, log=True):

    extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    image_paths = []

    for dir_path, _, files in os.walk(image_dir):
        for file in files:
            if any (file.lower().endswith(ext) for ext in extensions):
                image_paths.append(os.path.join(dir_path, file))

    if log:
        print(f"Total high resolution images: {len(image_paths)}")

    return image_paths

def create_datasets(train_path, val_path, hr_size, lr_size, batch_size, log=True):

    train_dataset = SRDataset(
        train_path, batch_size=batch_size, hr_size=hr_size, lr_size=lr_size, do_flip=True, rotate=True, mode='train'
        )
    
    val_dataset = SRDataset(
        val_path, batch_size=1, hr_size=hr_size, lr_size=lr_size, do_flip=False, rotate=False, mode='val'
        ) 

    train_steps = len(train_dataset)
    val_steps = len(val_dataset)      

    if log:
        print(f"Training batches: {len(train_dataset)} batches")
        print(f"Validation batches: {len(val_dataset)} batches")
    
    return train_dataset, val_dataset, train_steps, val_steps

class SRDataset(Sequence):
    """
    Custom dataset class which gives pair of HR and LR images, it also supports various data augmentations for training
    like random crop patches, flip images horizontally or vertically, random image rotation and 
    returns normalized arrays [0,1]
    """
    def __init__(self, image_paths, batch_size, hr_size=256, lr_size=64, val_size=(1356,2040), do_flip=False, rotate=False, mode="train"):  
        self.hr_images = image_paths
        self.batch_size = batch_size
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.val_size = val_size
        self.flip = do_flip
        self.rotate = rotate
        self.mode = mode

    def __len__(self):
        return len(self.hr_images) // self.batch_size

    def __getitem__(self, idx):

        batch_start = idx * self.batch_size
        batch_end = (idx + 1) * self.batch_size
        batch_path = self.hr_images[batch_start:batch_end]

        hr_batch, lr_batch = [], []
        for i, img_path in enumerate(batch_path):

            hr = Image.open(img_path).convert('RGB')

            if self.mode == "train":
                
                hr, lr = crop_patches(hr, self.hr_size)
                if self.flip:
                    hr, lr = flip_images(hr, lr)
                if self.rotate:
                    hr, lr = rotate_images(hr, lr)

            else:
                hr = hr.resize(self.val_size)
                w, h = hr.size
                lr = hr.resize((w//4, h//4),Image.BICUBIC)
                
            hr_batch.append(hr)
            lr_batch.append(lr)
            
        hr_batch = np.array(hr_batch, dtype=np.float32) / 255.0
        lr_batch = np.array(lr_batch, dtype=np.float32) / 255.0

        return hr_batch, lr_batch

    def on_epoch_end(self):
        if self.mode == 'train':
            np.random.shuffle(self.hr_images)