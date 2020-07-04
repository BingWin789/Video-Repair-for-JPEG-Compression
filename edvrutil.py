import tensorflow as tf
from residualutil import residual_block
from basicop import deform_conv_op, resize_bilinear, base_conv2d

def pcd_feat_extract(inputs, filters, convfcn, trainable=True, reuse=False, namescope='pcd_feat_extract'):
    """An Encoder, which is used to extract features in different scales."""
    with tf.variable_scope(namescope, reuse=reuse):
        # L1 feat
        L1_fea = convfcn(inputs, filters, trainable=trainable, namescope='l1_fea')
        L1_fea = tf.nn.leaky_relu(L1_fea)
        L1_fea = residual_block(L1_fea, filters, 3, convfcn, False, 1, 1, trainable=trainable, activation=tf.nn.leaky_relu, namescope='l1_fea_res')
        # L2 feat
        L2_fea = convfcn(L1_fea, filters, k_size=3, strides=2, trainable=trainable, namescope='l2_fea')
        L2_fea = tf.nn.leaky_relu(L2_fea)
        L2_fea = convfcn(L2_fea, filters, k_size=3, strides=1, trainable=trainable, namescope='l2_fea_2')
        L2_fea = tf.nn.leaky_relu(L2_fea)
        # L3 feat
        L3_fea = convfcn(L2_fea, filters, k_size=3, strides=2, trainable=trainable, namescope='l3_fea')
        L3_fea = tf.nn.leaky_relu(L3_fea)
        L3_fea = convfcn(L3_fea, filters, k_size=3, strides=1, trainable=trainable, namescope='l3_fea_2')
        L3_fea = tf.nn.leaky_relu(L3_fea)
    return L1_fea, L2_fea, L3_fea

def pcd_align(ref_fea_l1, 
            ref_fea_l2, 
            ref_fea_l3, 
            nbr_fea_l1, 
            nbr_fea_l2, 
            nbr_fea_l3, 
            k_size=3, 
            convfcn=base_conv2d, 
            trainable=True, 
            reuse=False, 
            namescope='pcd_align'):
    """this is a simple implementation of PCD module in edvr paper."""
    with tf.variable_scope(namescope, reuse=reuse):
        # align coarsest(L3) feature map first
        offset_prev_f3 = convfcn(tf.concat([ref_fea_l3, nbr_fea_l3], axis=1), filters=2 * k_size**2, 
                                        k_size=k_size, use_bias=True, trainable=trainable, namescope='prev_f3_offset')
        offset_prev_f3 = convfcn(offset_prev_f3, filters=2 * k_size**2, k_size=k_size, use_bias=True, 
                                        trainable=trainable, namescope='prev_f3_offset2')
        C = nbr_fea_l3.get_shape().as_list()[1]
        kernel = tf.get_variable(name='prev_f3_d_kernel', shape=(C, C, k_size, k_size),
                                 initializer=tf.keras.initializers.he_normal(), trainable=trainable)
        aligned_prev_f3 = deform_conv_op(nbr_fea_l3, filter=kernel, offset=offset_prev_f3, rates=[1, 1, 1, 1], padding='SAME',
                             strides=[1, 1, 1, 1], num_groups=1, deformable_group=1)
        
        # upsample aligned L3 feature map
        h, w = offset_prev_f3.get_shape().as_list()[2:4]
        offset_prev_f3_x2 = resize_bilinear(offset_prev_f3, (h * 2, w * 2)) * 2

        h, w = aligned_prev_f3.get_shape().as_list()[2:4]
        aligned_prev_f3_x2 = resize_bilinear(aligned_prev_f3, (h * 2, w * 2))

        # align L2 feature map
        offset_prev_f2 = convfcn(tf.concat([ref_fea_l2, nbr_fea_l2], axis=1), filters=2 * k_size**2, 
                                        k_size=k_size, use_bias=True, trainable=trainable, namescope='prev_f2_offset')
        offset_prev_f2 = convfcn(tf.concat([offset_prev_f2, offset_prev_f3_x2], axis=1), filters=2 * k_size**2, k_size=k_size, 
                                        use_bias=True, trainable=trainable, namescope='prev_f2_offset_2')
        offset_prev_f2 = convfcn(offset_prev_f2, filters=2 * k_size**2, k_size=k_size, use_bias=True, trainable=trainable, namescope='prev_f2_offset_3')
        C = nbr_fea_l2.get_shape().as_list()[1]
        kernel = tf.get_variable(name='prev_f2_d_kernel', shape=(C, C, k_size, k_size),
                                 initializer=tf.keras.initializers.he_normal(), trainable=trainable)
        aligned_prev_f2 = deform_conv_op(nbr_fea_l2, filter=kernel, offset=offset_prev_f2, rates=[1, 1, 1, 1], padding='SAME',
                             strides=[1, 1, 1, 1], num_groups=1, deformable_group=1)
        aligned_prev_f2 = convfcn(tf.concat([aligned_prev_f2, aligned_prev_f3_x2], axis=1), C, k_size, trainable=trainable, namescope='cmb_prev_curr_2')
        
        # upsample aligned L2 feature map
        h, w = offset_prev_f2.get_shape().as_list()[2:4]
        offset_prev_f2_x2 = resize_bilinear(offset_prev_f2, (h * 2, w * 2)) * 2

        h, w = aligned_prev_f2.get_shape().as_list()[2:4]
        aligned_prev_f2_x2 = resize_bilinear(aligned_prev_f2, (h * 2, w * 2))


        # align L1 feature map
        offset_prev_f1 = convfcn(tf.concat([ref_fea_l1, nbr_fea_l1], axis=1), filters=2 * k_size**2, 
                                        k_size=k_size, use_bias=True, trainable=trainable, namescope='prev_f1_offset')
        offset_prev_f1 = convfcn(tf.concat([offset_prev_f1, offset_prev_f2_x2], axis=1), filters=2 * k_size**2, 
                                        k_size=k_size, use_bias=True, trainable=trainable, namescope='prev_f1_offset_2')
        offset_prev_f1 = convfcn(offset_prev_f1, filters=2 * k_size**2, k_size=k_size, use_bias=True, trainable=trainable, namescope='prev_f1_offset_3')
        C = nbr_fea_l1.get_shape().as_list()[1]
        kernel = tf.get_variable(name='prev_f1_d_kernel', shape=(C, C, k_size, k_size),
                                 initializer=tf.keras.initializers.he_normal(), trainable=trainable)
        aligned_prev_f1 = deform_conv_op(nbr_fea_l1, filter=kernel, offset=offset_prev_f1, rates=[1, 1, 1, 1], padding='SAME',
                             strides=[1, 1, 1, 1], num_groups=1, deformable_group=1)
        aligned_prev_f1 = convfcn(tf.concat([aligned_prev_f1, aligned_prev_f2_x2], axis=1), C, k_size, trainable=trainable, namescope='cmb_prev_curr_1')

        # refine aligned L1 feature map
        offset = convfcn(tf.concat([ref_fea_l1, aligned_prev_f1], axis=1), filters=2 * k_size**2, k_size=k_size, use_bias=True, trainable=trainable, namescope='offset')
        offset = convfcn(offset, filters=2 * k_size**2, k_size=k_size, use_bias=True, trainable=trainable, namescope='offset_2')
        C = aligned_prev_f1.get_shape().as_list()[1]
        kernel = tf.get_variable(name='refine_d_kernel', shape=(C, C, k_size, k_size),
                                 initializer=tf.keras.initializers.he_normal(), trainable=trainable)
        aligned_f1_feat = deform_conv_op(aligned_prev_f1, filter=kernel, offset=offset, rates=[1, 1, 1, 1], padding='SAME',
                             strides=[1, 1, 1, 1], num_groups=1, deformable_group=1)

    return aligned_f1_feat

def tsa_fusion(inputs, filters, convfcn=base_conv2d, batch_size=32, trainable=True, namescope='tsa_fusion'):
    """this is tensorflow implementation of official pytorch vesion for TSA fusion module."""
    with tf.variable_scope(namescope):
        inputs_pre = inputs
        _, N, C, H, W = inputs.get_shape().as_list()  # N: number of neighbour frames.
        center = N // 2
        #### temporal attention
        emb_ref = convfcn(inputs[:, center, :, :, :], filters, 3, trainable=trainable, namescope='tAtt_2')
        emb = tf.reshape(inputs, [-1, C, H, W])
        emb = convfcn(emb, filters, 3, trainable=trainable, namescope='tAtt_1')
        emb = tf.reshape(emb, [batch_size, N, -1, H, W])

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = tf.reduce_sum(emb_nbr * emb_ref, axis=1, keepdims=True)
            cor_l.append(cor_tmp)
        cor_prob = tf.sigmoid(tf.concat(cor_l, axis=1)) 
        cor_prob = tf.expand_dims(cor_prob, axis=2)
        cor_prob = tf.tile(cor_prob, [1, 1, C, 1, 1])
        cor_prob = tf.reshape(cor_prob, [batch_size, -1, H, W])
        inp = tf.reshape(inputs, [batch_size, -1, H, W])
        inputs = inp * cor_prob
        

        #### fusion
        fea = convfcn(inputs, filters, 1, trainable=trainable, namescope='fea_fusion')
        fea = tf.nn.leaky_relu(fea)

        #### spatial attention
        att = convfcn(inputs, filters, 1, trainable=trainable, namescope='sAtt_1')
        att = tf.nn.leaky_relu(att)
        att_max = tf.nn.max_pool(att, (1, 1, 3, 3), (1, 1, 2, 2), padding='SAME', data_format='NCHW')
        att_avg = tf.nn.avg_pool(att, (1, 1, 3, 3), (1, 1, 2, 2), padding='SAME', data_format='NCHW')
        att = convfcn(tf.concat([att_max, att_avg], axis=1), filters, 1, trainable=trainable, namescope='sAtt_2')
        att = tf.nn.leaky_relu(att)
        # pyramid levels
        att_L = convfcn(att, filters, 1, trainable=trainable, namescope='sAtt_L1')
        att_L = tf.nn.leaky_relu(att_L)
        att_max = tf.nn.max_pool(att_L, (1, 1, 3, 3), (1, 1, 2, 2), padding='SAME', data_format='NCHW')
        att_avg = tf.nn.avg_pool(att_L, (1, 1, 3, 3), (1, 1, 2, 2), padding='SAME', data_format='NCHW')
        att_L = convfcn(tf.concat([att_max, att_avg], axis=1), filters, 3, trainable=trainable, namescope='sAtt_L2')
        att_L = tf.nn.leaky_relu(att_L)
        att_L = convfcn(att_L, filters, 3, trainable=trainable, namescope='sAtt_L3')
        att_L = tf.nn.leaky_relu(att_L)
        h, w = att_L.get_shape().as_list()[2:4]
        att_L = resize_bilinear(att_L, (h*2, w*2), align_corners=True)

        att = convfcn(att, filters, 3, trainable=trainable, namescope='sAtt_3')
        att = tf.nn.leaky_relu(att)
        att += att_L
        att = convfcn(att, filters, 1, trainable=trainable, namescope='sAtt_4')
        att = tf.nn.leaky_relu(att)
        h, w = att.get_shape().as_list()[2:4]
        att = resize_bilinear(att, (h*2, w*2), align_corners=True)
        att = convfcn(att, filters, 3, trainable=trainable, namescope='sAtt_5')
        att_add = convfcn(att, filters, 1, trainable=trainable, namescope='sAtt_add_1')
        att_add = tf.nn.leaky_relu(att_add)
        att_add = convfcn(att_add, filters, 1, trainable=trainable, namescope='sAtt_add_2')
        att = tf.nn.sigmoid(att)

        # ref: https://cloud.tencent.com/developer/article/1517589
        # 3d conv for inputs_pre B N C H W -> B C N H W
        inputs_pre = tf.transpose(inputs_pre, [0, 2, 1, 3, 4])
        inputs_pre = tf.layers.conv3d(inputs_pre, filters, 3, 1, 'SAME', 'channels_first', kernel_initializer=tf.keras.initializers.he_normal())
        inputs_pre = tf.transpose(inputs_pre, [0, 2, 1, 3, 4])
        inputs_pre = tf.reshape(inputs_pre, [batch_size, N * C, H, W])
        inputs_pre = convfcn(inputs_pre, filters, 1, trainable=trainable, namescope='fea_fusion2')        

        fea += inputs_pre

        fea = fea * att * 2 + att_add
    return fea