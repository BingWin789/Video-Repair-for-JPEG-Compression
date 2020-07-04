import os
import tensorflow as tf
import numpy as np
import time

# set random seed
tf.set_random_seed(100)

from metrics import psnr_metric, ssim_metric
from losses import charbonnier_penalty_loss
from dataset import Dataset
from model import model
from basicop import to_channels_last
from params import trainargs, modelargs, datasetargs

def _get_model_init_fn():
    return None  

def _float32_to_uint8(images):
    images = images * 255.0
    images = tf.round(images)
    images = tf.saturate_cast(images, tf.uint8)
    return images

def _build_model(iterator):
    # extract train data
    samples = iterator.get_next()
    curr_frame = samples['input']
    labels = samples['label']
    # model
    curr_frame_inp = tf.transpose(curr_frame, [0, 1, 4, 2, 3])
    preds_lst = model(curr_frame_inp, modelargs)
    preds1 = to_channels_last(preds_lst)
    # loss
    ls1_abs = charbonnier_penalty_loss(preds1, labels)
    tf.summary.scalar('loss/absdiff', ls1_abs)
    # summaries
    tf.summary.image('img_0_labl', _float32_to_uint8(labels[0:4, :, :, :]))
    tf.summary.image('img_3_pred1', _float32_to_uint8(preds1[0:4, :, :, :]))
    tf.summary.image('img_4_curr', _float32_to_uint8(curr_frame[0:4, 2, :, :, :]))
    # metrics
    psnrval = psnr_metric(preds1, labels)
    tf.summary.scalar('metric/psnr', psnrval)
    ssimval = ssim_metric(preds1, labels)
    tf.summary.scalar('metric/ssim', ssimval)
    tf.summary.scalar('metric/total', psnrval + (ssimval - 0.4) / 0.6 * 50)


def _tower_loss(iterator, scope, reuse_variable):
    with tf.variable_scope(tf.get_variable_scope(), reuse=True if reuse_variable else None):
        _build_model(iterator)
    losses = tf.losses.get_losses(scope=scope)
    loss = tf.add_n(losses)
    return loss

def _average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads, variables = zip(*grad_and_vars)
        grads = tf.stack(grads, axis=0)
        grad = tf.reduce_mean(grads, axis=0)
        # All vars are of the same value, using the first tower here.
        average_grads.append((grad, variables[0]))
        
    return average_grads


def _train_model(iterator):
    global_step = tf.train.get_or_create_global_step()
    learning_rate = trainargs.learning_rate
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    tower_losses = []
    tower_grads = []
    for i in range(trainargs.num_gpus):
        with tf.device('/gpu:%d' % i):
            # First tower has default name scope.
            name_scope = ('clone_%d' % i) if i else ''
            with tf.name_scope(name_scope) as scope:
                loss = _tower_loss(
                    iterator=iterator,
                    scope=scope,
                    reuse_variable=(i != 0))
                tower_losses.append(loss)

    for i in range(trainargs.num_gpus):
        with tf.device('/gpu:%d' % i):
            name_scope = ('clone_%d' % i) if i else ''
            with tf.name_scope(name_scope) as scope:
                grads = optimizer.compute_gradients(tower_losses[i])
                tower_grads.append(grads)

    with tf.device('/cpu:0'):
        grads_and_vars = _average_gradients(tower_grads)
        # ))
        # Create gradient update op.
        grad_updates = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)
        
        # Gather update_ops. These contain, for example,
        # the updates for the batch_norm variables created by model_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)
        total_loss = tf.losses.get_total_loss(add_regularization_losses=False)
        tf.summary.scalar('loss/total', total_loss)

        with tf.control_dependencies([update_op]):
            train_tensor = tf.identity(total_loss, name='train_op')

        # Excludes summaries from towers other than the first one.
        summary_op = tf.summary.merge_all(scope='(?!clone_)')
    return train_tensor, summary_op

def main(unused_argv):
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(tf.train.replica_device_setter(ps_tasks=0)):
            dataset = Dataset('train', 
                            datasetargs.tfrecord_dir, 
                            datasetargs.image_dir,
                            trainargs.batch_size, 
                            trainargs.patch_size,
                            num_readers=8,
                            is_training=True,
                            should_shuffle=True,
                            should_repeat=True)
            train_tensor, summary_op = _train_model(dataset.get_one_shot_iterator())
            # Soft placement allows placing on CPU ops without GPU implementation.
            session_config = tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False)
            # session_config.gpu_options.allow_growth = True
            init_fn = _get_model_init_fn()
            scaffold = tf.train.Scaffold(
                init_fn=init_fn,
                summary_op=summary_op,
            )
            stop_hook = tf.train.StopAtStepHook(
                last_step=trainargs.training_number_of_steps)
            profile_dir = trainargs.profile_logdir
            if profile_dir is not None:
                tf.gfile.MakeDirs(profile_dir)
            with tf.contrib.tfprof.ProfileContext(
                enabled=profile_dir is not None, profile_dir=profile_dir):
                with tf.train.MonitoredTrainingSession(
                    master='',
                    is_chief=True,
                    config=session_config,
                    scaffold=scaffold,
                    checkpoint_dir=trainargs.train_logdir,
                    log_step_count_steps=trainargs.log_steps,
                    save_summaries_steps=trainargs.save_summaries_secs,
                    save_checkpoint_secs=trainargs.save_ckpt_secs,
                    hooks=[stop_hook]) as sess:
                        while not sess.should_stop():
                            print(sess.run(tf.train.get_or_create_global_step()))
                            sess.run([train_tensor])


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = trainargs.visuable_gpus
    trainargs.num_gpus = len(trainargs.visuable_gpus.split(','))

    def mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)
    
    mkdir(trainargs.train_logdir)
    mkdir(trainargs.profile_logdir)
    
    tf.app.run()
