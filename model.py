import tensorflow as tf
from edvrutil import pcd_feat_extract, pcd_align, tsa_fusion
from residualutil import residual_group
from basicop import base_conv2d, wn_conv2d

"""NOTE: channels first, namely [N, C, H, W] is used in all operations."""


def model(inputs, args, trainable=True, namescope='model'):
    """model is modified from edvr, and is used to repair video artifacts caused by jpeg compression.
    
    Args:
        inputs: damaged 5 consecutive frames. shape: (Batch size, N, Channels, Height, Weight). N = 5. The center frame 
            will be repaired.
    
    Returns:
        repaired center frame.
    """
    with tf.variable_scope(namescope):
        convfcn = wn_conv2d if args.weight_norm else base_conv2d  # use weight norm or not
        # extract reference frame feature
        ref_fea1, ref_fea2, ref_fea3 = pcd_feat_extract(inputs[:, 2, :, :, :], args.filters_num, convfcn, 
                                                            trainable=trainable, reuse=False, namescope='ref_pcd_fea')
        # extract neiboughr frame feature
        nbr0_fea1, nbr0_fea2, nbr0_fea3 = pcd_feat_extract(inputs[:, 0, :, :, :], args.filters_num, convfcn, 
                                                            trainable=trainable, reuse=False, namescope='nbr_pcd_fea')
        nbr1_fea1, nbr1_fea2, nbr1_fea3 = pcd_feat_extract(inputs[:, 1, :, :, :], args.filters_num, convfcn, 
                                                            trainable=trainable, reuse=True, namescope='nbr_pcd_fea')
        nbr3_fea1, nbr3_fea2, nbr3_fea3 = pcd_feat_extract(inputs[:, 3, :, :, :], args.filters_num, convfcn, 
                                                            trainable=trainable, reuse=True, namescope='nbr_pcd_fea')
        nbr4_fea1, nbr4_fea2, nbr4_fea3 = pcd_feat_extract(inputs[:, 4, :, :, :], args.filters_num, convfcn, 
                                                            trainable=trainable, reuse=True, namescope='nbr_pcd_fea')
        # align neiboughr features to reference frame
        aligned_nbr0_fea1 = pcd_align(ref_fea1, ref_fea2, ref_fea3, nbr0_fea1, nbr0_fea2, nbr0_fea3, 
                                        convfcn=convfcn, trainable=trainable, reuse=False, namescope='pcd_align')
        aligned_nbr1_fea1 = pcd_align(ref_fea1, ref_fea2, ref_fea3, nbr1_fea1, nbr1_fea2, nbr1_fea3, 
                                        convfcn=convfcn, trainable=trainable, reuse=True, namescope='pcd_align')
        aligned_nbr3_fea1 = pcd_align(ref_fea1, ref_fea2, ref_fea3, nbr3_fea1, nbr3_fea2, nbr3_fea3, 
                                        convfcn=convfcn, trainable=trainable, reuse=True, namescope='pcd_align')
        aligned_nbr4_fea1 = pcd_align(ref_fea1, ref_fea2, ref_fea3, nbr4_fea1, nbr4_fea2, nbr4_fea3, 
                                        convfcn=convfcn, trainable=trainable, reuse=True, namescope='pcd_align')
        # fuse aligned neibour features and reference feature
        fusion_fea = tf.stack([aligned_nbr0_fea1, aligned_nbr1_fea1, ref_fea1, aligned_nbr3_fea1, aligned_nbr4_fea1], axis=1)        
        fusion_fea = tsa_fusion(fusion_fea, args.filters_num, convfcn, args.batch_size, trainable=trainable)

        # start to repair.
        with tf.variable_scope('generator'):
            x = convfcn(fusion_fea, args.filters_num, 3, trainable=trainable, namescope='input')
            x = tf.nn.leaky_relu(x)

            for i in range(args.groups_num):
                x = residual_group(x, args.filters_num, 3, convfcn, args.blocks_num, args.expand, args.use_channel_attention, 
                                        args.channel_att_reduction, trainable, namescope='group_%d'%i)

            x = convfcn(x, args.filters_num, 3, trainable=trainable, namescope='fuse')
            x = convfcn(x, args.filters_num//2, 3, trainable=trainable, namescope='down')
            x = convfcn(x, 3, 3, trainable=trainable, namescope='output')

    return x