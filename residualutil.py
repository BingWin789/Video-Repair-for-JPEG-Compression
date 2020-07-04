import tensorflow as tf

def channel_attention(inputs, convfcn, reduction=16, trainable=True, namescope='calayer'):
    """channel attention is referenced from RCAN paper."""
    with tf.variable_scope(namescope):
        _, c, h, w = inputs.get_shape().as_list()
        avg = tf.nn.avg_pool(inputs, ksize=(1, 1, h, w), strides=(1, 1, h, w), padding='VALID', data_format='NCHW')
        conv_dn = convfcn(avg, c//reduction, 1, padding='VALID', use_bias=True, trainable=trainable, namescope='down')
        conv_dn = tf.nn.relu(conv_dn)
        conv_up = convfcn(conv_dn, c, 1, padding='VALID', use_bias=True, trainable=trainable, namescope='up')
        conv_up = tf.nn.sigmoid(conv_up)
        result = inputs * conv_up
    return result

def residual_block(inputs, 
                    filters, 
                    k_size, 
                    convfcn, 
                    channelatt, 
                    reduction, 
                    expand=1, 
                    trainable=True, 
                    activation=tf.nn.relu, 
                    namescope='residual_block'):
    """basic resdual block with channel attention."""
    with tf.variable_scope(namescope):
        skip = inputs
        x = convfcn(inputs, filters * expand, k_size, trainable=trainable, namescope='conv1')
        x = activation(x)
        x = convfcn(x, filters, k_size, trainable=trainable, namescope='conv2')
        if channelatt:
            x = channel_attention(x, convfcn, reduction=reduction, trainable=trainable)
        x += skip
    return x

def residual_group(inputs, 
                    filters, 
                    k_size, 
                    convfcn, 
                    n_blocks, 
                    expand, 
                    channelatt=False, 
                    reduction=8, 
                    trainable=True, 
                    namescope='residual_group'):
    """A group of residual blocks connected in series with dense connections."""
    with tf.variable_scope(namescope):
        skip = inputs
        x = convfcn(inputs, filters, 3, trainable=trainable, namescope='input')

        imdt_out = []
        for i in range(n_blocks):
            x = residual_block(x, filters, k_size, convfcn, channelatt, reduction, expand, trainable, namescope='block_%d'%i)
            imdt_out.append(x)

        x = tf.concat(imdt_out, axis=1)
        x = convfcn(x, filters, 1, trainable=trainable, namescope='combine')
        x = convfcn(x, filters, 3, trainable=trainable, namescope='endconv')

        x += skip
    return x
