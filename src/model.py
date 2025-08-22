import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, Input, Lambda
from tensorflow.keras.models import Model

def residual_block(x, num_channels):

    out = Conv2D(num_channels, kernel_size=3, padding='same', activation='relu')(x)
    out = Conv2D(num_channels, kernel_size=3, padding='same')(out)
    res = Add()([out, x])
    return res

def num_residual_blocks(x, num_channels, num_blocks):

    for _ in range(num_blocks):
        x = residual_block(x, num_channels)
    return x

def pixel_shuffle(scale):

    """
    scale: Integer scaling factor (e.g., 2 for 2x upscaling).
    Returns:
        Lambda layer that upsamples input from (H, W, C * scale^2) to (scale*H, scale*W, C)
    """
    def upscale(x):
        return tf.nn.depth_to_space(x, scale)
    return Lambda(upscale)

def edsr(num_channels=64, scale_factor=2, num_blocks=16):

    input_image = Input(shape=(None, None, 3), name='input')

    # First convolution
    conv_1 = Conv2D(num_channels, kernel_size=3, padding='same', name='Conv_initial')(input_image)

    # Residual blocks
    res = num_residual_blocks(conv_1, num_channels, num_blocks)

    # Convolution before upsampling + skip connection
    conv_2 = Conv2D(num_channels, kernel_size=3, padding='same', name='Conv_pre_upsampling')(res)
    add = Add(name='Skip_connection')([conv_2, conv_1])

    # Upsampling with pixel shuffle
    upsample = add
    if scale_factor in [2, 4]:
        upsample = Conv2D(num_channels * (2 ** 2), kernel_size=3, padding='same')(upsample)
        upsample = pixel_shuffle(2)(upsample)
    if scale_factor == 4:
        upsample = Conv2D(num_channels * (2 ** 2), kernel_size=3, padding='same')(upsample)
        upsample = pixel_shuffle(2)(upsample)

    # Output layer
    out = Conv2D(3, kernel_size=3, padding='same', name='Output')(upsample)

    model = Model(inputs=input_image, outputs=out, name=f'EDSR_x{scale_factor}')

    return model

if __name__ == "__main__":

    model = edsr(num_channels=64, scale_factor=4, num_blocks=16)
    model.summary()