import tensorflow as tf

def psnr_metric(preds, labels, namescope='psnr'):
    def _float32_to_uint8(images):
        images = images * 255.0
        images = tf.round(images)
        images = tf.saturate_cast(images, tf.uint8)
        return images

    val = tf.image.psnr(
        _float32_to_uint8(labels), 
        _float32_to_uint8(preds),
        max_val=255,
    )
    return tf.reduce_mean(val, name=namescope)

def ssim_metric(preds, labels, namescope='ssim'):
    return tf.reduce_mean(tf.image.ssim(preds, labels, max_val=1.0), name=namescope)

def total_metric(preds, labels, namescope='metric'):
    psnr = psnr_metric(preds, labels)
    ssim = ssim_metric(preds, labels)
    res = psnr + (ssim - 0.4) / 0.6 * 50.0
    res = tf.multiply(res, 1.0, name=namescope)
    return res
