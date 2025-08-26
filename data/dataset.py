import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import Sequence
from .augmentations import crop_patches, flip_images

def create_data_splits(hr_image_dir, val_size=0.2, test_size=0.1, random_state=41, log=True):

    extensions = ['jpg', 'jpeg', 'png', 'bmp']
    image_paths = []

    for dir_path, _, files in os.walk(hr_image_dir):
        for file in files:
            if any (file.lower().endswith(ext) for ext in extensions):
                image_paths.append(os.path.join(dir_path, file))

    if log:
        print(f"Total high resolution images: {len(image_paths)}")

    train_val_paths, test_path = train_test_split(image_paths, test_size=test_size, random_state=random_state, shuffle=True)

    train_path, val_path = train_test_split(train_val_paths, test_size=val_size, random_state=random_state, shuffle=True)

    if log:
        print(f"Number of training images: {len(train_path)}")
        print(f"Number of validation images: {len(val_path)}")
        print(f"Number of test images: {len(test_path)}")

    return train_path, val_path, test_path

def create_datasets(train_path, val_path, test_path, hr_size, lr_size, batch_size, log=True):

    train_dataset = SRDataset(
        train_path, batch_size=batch_size, hr_size=hr_size, lr_size=lr_size, do_flip=True, mode='train'
        )
    
    val_dataset = SRDataset(
        val_path, batch_size=batch_size, hr_size=hr_size, lr_size=lr_size, do_flip=False, mode='val'
        )
    
    test_dataset = SRDataset(
        test_path, batch_size=batch_size, hr_size=hr_size, lr_size=lr_size, do_flip=False, mode='test'
        )        

    if log:
        print(f"Training batches: {len(train_dataset)}")
        print(f"Validation batches: {len(val_dataset)}")
        print(f"Test batches: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset

class SRDataset(Sequence):

    def __init__(self, image_paths, batch_size, hr_size=256, lr_size=64, val_size=(1200,800), do_flip=False, mode="train"):  
        self.hr_images = image_paths
        self.batch_size = batch_size
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.val_size = val_size
        self.flip = do_flip
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