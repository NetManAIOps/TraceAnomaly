# -*- coding: utf-8 -*-
import sys
from argparse import ArgumentParser

import tensorflow as tf
from pprint import pformat
from tensorflow.contrib.framework import arg_scope

import tfsnippet as spt
from tfsnippet.examples.utils import MLResults, print_with_title


class ExpConfig(spt.Config):
    # model parameters
    x_dim = 784
    l2_reg = 0.0001

    # training parameters
    result_dir = None
    write_summary = False
    max_epoch = 500
    max_step = None
    batch_size = 64
    test_batch_size = 256

    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 50
    lr_anneal_step_freq = None


config = ExpConfig()


@spt.global_reuse
def model(x):
    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_x = x
        h_x = spt.layers.dense(h_x, 500)
        h_x = spt.layers.dense(h_x, 500)
    logits = spt.layers.dense(h_x, 10, name='logits')
    return logits


def main():
    # parse the arguments
    arg_parser = ArgumentParser()
    spt.register_config_arguments(config, arg_parser)
    arg_parser.parse_args(sys.argv[1:])

    # print the config
    print_with_title('Configurations', pformat(config.to_dict()), after='\n')

    # open the result object and prepare for result directories
    results = MLResults(config.result_dir)
    results.save_config(config)  # save experiment settings for review
    results.make_dirs('train_summary', exist_ok=True)

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.float32, shape=(None, config.x_dim), name='input_x')
    input_y = tf.placeholder(
        dtype=tf.int32, shape=[None], name='input_y')
    learning_rate = spt.AnnealingVariable(
        'learning_rate', config.initial_lr, config.lr_anneal_factor)

    # derive the loss, output and accuracy
    logits = model(input_x)
    cls_loss = tf.losses.sparse_softmax_cross_entropy(input_y, logits)
    loss = cls_loss + tf.losses.get_regularization_loss()
    y = spt.ops.softmax_classification_output(logits)
    acc = spt.ops.classification_accuracy(y, input_y)

    # derive the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    params = tf.trainable_variables()
    grads = optimizer.compute_gradients(loss, var_list=params)
    with tf.control_dependencies(
            tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.apply_gradients(grads)

    # prepare for training and testing data
    (x_train, y_train), (x_test, y_test) = \
        spt.datasets.load_mnist(normalize_x=True)
    train_flow = spt.DataFlow.arrays([x_train, y_train], config.batch_size,
                                     shuffle=True, skip_incomplete=True)
    test_flow = spt.DataFlow.arrays([x_test, y_test], config.test_batch_size)

    with spt.utils.create_session().as_default(), \
            train_flow.threaded(5) as train_flow:
        # train the network
        with spt.TrainLoop(params,
                           max_epoch=config.max_epoch,
                           max_step=config.max_step,
                           summary_dir=(results.system_path('train_summary')
                                        if config.write_summary else None),
                           summary_graph=tf.get_default_graph(),
                           early_stopping=False) as loop:
            trainer = spt.Trainer(
                loop, train_op, [input_x, input_y], train_flow,
                metrics={'loss': loss, 'acc': acc}
            )
            trainer.anneal_after(
                learning_rate,
                epochs=config.lr_anneal_epoch_freq,
                steps=config.lr_anneal_step_freq
            )
            evaluator = spt.Evaluator(
                loop,
                metrics={'test_acc': acc},
                inputs=[input_x, input_y],
                data_flow=test_flow,
                time_metric_name='test_time'
            )
            evaluator.after_run.add_hook(
                lambda: results.update_metrics(evaluator.last_metrics_dict))
            trainer.evaluate_after_epochs(evaluator, freq=5)
            trainer.log_after_epochs(freq=1)
            trainer.run()

    # print the final metrics and close the results object
    print_with_title('Results', results.format_metrics(), before='\n')
    results.close()


if __name__ == '__main__':
    main()
