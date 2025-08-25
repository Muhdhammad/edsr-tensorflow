import os
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.saving import register_keras_serializable

def inference(input_image, model_path, show_comparision=False, save_path=None):

    # Register the custom 'upscale' function
    @register_keras_serializable()
    def upscale(x):
        return tf.nn.depth_to_space(x, block_size=2)

    model_path = str(model_path)

    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'upscale':upscale}, compile=False)
    except Exception as e:
        raise Exception(f"Failed to load model from {model_path}: {e}")
    
    # ===== Load and Preprocessing =====
    img = Image.open(str(input_image)).convert('RGB')
    img_arr = np.array(img, dtype=np.float32) / 255
    img_input = np.expand_dims(img_arr, axis=0)

    # ===== Inference ======
    sr_img = model.predict(img_input, verbose=0)

    # ===== Postprocessing =====
    sr_image = (np.clip(sr_img, 0, 1) * 255).astype(np.uint8)
    sr_image = sr_image[0] # H x W x C

    # ===== visualization =====
    if show_comparision:

        import matplotlib.pyplot as plt

        fig_img, fig_axs = plt.subplots(1, 2, figsize=(18, 12))

        fig_axs[0].imshow(img_arr)
        fig_axs[0].set_title("Input - Low Resolution image")
        fig_axs[0].axis("off")

        fig_axs[1].imshow(sr_image)
        fig_axs[1].set_title("Output - Super Resolution image")
        fig_axs[1].axis("off")

        plt.show()

    # ===== save the image if requested =====
    if save_path:
        Image.fromarray(sr_image).save(save_path)

    return sr_image