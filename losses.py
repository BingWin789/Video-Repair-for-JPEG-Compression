import tensorflow as tf

def abs_diff_loss(preds, labels, weights=1.0, namescope='absdiff'):
    loss = tf.losses.absolute_difference(labels=labels, predictions=preds, weights=weights, scope=namescope)
    return loss

def ssim_loss(preds, labels, weights=1.0, namescope='ssimloss'):
    _ssim = 1.0 - tf.image.ssim(preds, labels, max_val=1.0)
    loss = tf.divide(tf.reduce_sum(_ssim), tf.maximum(tf.reduce_sum(tf.to_float(tf.not_equal(_ssim, 0.0))), 1.0), name=namescope) * weights
    tf.losses.add_loss(loss)
    return loss

def charbonnier_penalty_loss(preds, labels):
    epsilon = 1e-3
    loss = tf.squared_difference(preds, labels) + epsilon**2
    loss = tf.sqrt(loss)
    loss = tf.reduce_mean(loss)
    tf.losses.add_loss(loss)
    return loss
