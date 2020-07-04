import tensorflow as tf
from lib.deform_conv_op import deform_conv_op

def to_channels_first(inputs):
    return tf.transpose(inputs, [0, 3, 1, 2])

def to_channels_last(inputs):
    return tf.transpose(inputs, [0, 2, 3, 1])

def resize_bilinear(inputs, size, align_corners=True):
    x = to_channels_last(inputs)
    x = tf.image.resize_bilinear(x, size, align_corners=align_corners)
    x = to_channels_first(x)
    return x

class Conv2DWeightNorm(tf.layers.Conv2D):
    def build(self, input_shape):
        self.wn_g = self.add_variable(  # add_weight -> add_variable
            name='wn_g',
            shape=(self.filters,),
            dtype=self.dtype,
            initializer=tf.initializers.ones,
            trainable=True,
        )
        super(Conv2DWeightNorm, self).build(input_shape)
        square_sum = tf.reduce_sum(
            tf.square(self.kernel), [0, 1, 2], keepdims=False)
        inv_norm = tf.rsqrt(square_sum)
        self.kernel = self.kernel * (inv_norm * self.wn_g)


def wn_conv2d(inputs,
            filters,
            k_size=3,
            strides=(1, 1),
            dilation_rate=(1, 1),
            padding='same',
            use_bias=True,
            trainable=True,
            namescope='wn_conv2d',
            data_format='channels_first',
            activation=None,
            kernel_initializer=None,
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            reuse=None):
    """this is weight normalization convolution, implementation for WDSR paper."""
    layer = Conv2DWeightNorm(
        filters=filters,
        kernel_size=k_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=namescope,
        dtype=inputs.dtype.base_dtype,
        _reuse=reuse,
        _scope=namescope)
    res = layer.apply(inputs)
    return res

def base_conv2d(inputs, 
                filters, 
                k_size=3, 
                strides=1, 
                dilation_rate=1, 
                padding='SAME',
                use_bias=True, 
                trainable=True, 
                namescope='base_conv2d'):
    with tf.variable_scope(namescope):
        x = tf.layers.conv2d(inputs, filters, k_size, strides, padding=padding, 
                    dilation_rate=dilation_rate, use_bias=use_bias, trainable=trainable, 
                    data_format='channels_first', 
                    kernel_initializer=tf.keras.initializers.he_normal())
    return x