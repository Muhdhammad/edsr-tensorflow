import tensorflow as tf

def calculate_psnr(sr, hr, eps=1e-10):
    mse = tf.reduce_mean(tf.square(sr - hr), axis=[1,2,3])   # H x W x C
    return 20 * tf.math.log(1.0 / tf.math.sqrt(mse + eps)) / tf.math.log(10.0)

def calculate_ssim(sr, hr):
    return tf.image.ssim(sr, hr, max_val=1.0)

def charbonnier_loss(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-3)))

