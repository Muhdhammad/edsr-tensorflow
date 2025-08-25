import random
from PIL import Image

def crop_patches(hr, patch_size=256):

    patch_size = patch_size
    lr_patch_size = (patch_size // 4, patch_size // 4)
 
    w, h = hr.size

    left = random.randrange(0, w-patch_size)
    top =  random.randrange(0, h-patch_size)

    right = left + patch_size
    bottom = top + patch_size

    hr_patch = hr.crop((left, top, right, bottom))

    lr_patch = hr_patch.resize(lr_patch_size , Image.BICUBIC)

    return hr_patch, lr_patch

def flip_images(hr, lr):

    hr_img, lr_img = hr, lr
    # flip image horizontally
    if random.random() > 0.5:
        hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
        lr_img = lr_img.transpose(Image.FLIP_LEFT_RIGHT)
        
    # flip image vertically
    if random.random() > 0.5:
        hr_img = hr_img.transpose(Image.FLIP_TOP_BOTTOM)
        lr_img = lr_img.transpose(Image.FLIP_TOP_BOTTOM)

    return hr_img, lr_img