import os

WORKDIR = os.path.dirname(os.path.abspath(__file__))

class DatasetArgs():
    '''parameters used for dataset'''
    pass

datasetargs = DatasetArgs()
datasetargs.image_dir = os.path.join(WORKDIR, 'image')
datasetargs.damage_sub_dir = 'damaged'
datasetargs.label_sub_dir = 'label'
datasetargs.tfrecord_dir = os.path.join(WORKDIR, 'tfrecord')
datasetargs.pkl_path = os.path.join(WORKDIR, 'image', 'traindata.pkl')


class TrainArgs():
    '''parameters used for train process.'''
    pass

trainargs = TrainArgs()
trainargs.patch_size = 96
trainargs.learning_rate = 1e-4
trainargs.batch_size = 16
trainargs.training_number_of_steps = 120000
trainargs.train_logdir = os.path.join(WORKDIR, 'ckpts')
trainargs.log_steps = 1000
trainargs.save_ckpt_secs = 600
trainargs.save_summaries_secs = 120
trainargs.profile_logdir = os.path.join(WORKDIR, 'profile')
trainargs.visuable_gpus = '0'  # use '0, 1, 2' for multi-gpu case


class ModelArgs():
    '''parameters used for model configuration.'''
    pass

modelargs = ModelArgs()
modelargs.use_channel_attention = True
modelargs.channel_att_reduction = 8
modelargs.filters_num = 64
modelargs.blocks_num = 5
modelargs.groups_num = 3
modelargs.weight_norm = False
modelargs.expand = 4
modelargs.batch_size = trainargs.batch_size
