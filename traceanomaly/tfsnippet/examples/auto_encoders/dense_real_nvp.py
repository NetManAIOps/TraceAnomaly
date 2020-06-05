# -*- coding: utf-8 -*-
import functools
import sys
from argparse import ArgumentParser

import tensorflow as tf
from pprint import pformat
from tensorflow.contrib.framework import arg_scope, add_arg_scope

import tfsnippet as spt
from tfsnippet.examples.utils import (MLResults,
                                      save_images_collection,
                                      bernoulli_as_pixel,
                                      bernoulli_flow,
                                      print_with_title)


class ExpConfig(spt.Config):
    # model parameters
    z_dim = 40
    x_dim = 784
    n_flows = 10
    n_flow_hidden_layers = 1

    # training parameters
    result_dir = None
    write_summary = False
    max_epoch = 3000
    max_step = None
    batch_size = 128
    l2_reg = 0.0001
    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 300
    lr_anneal_step_freq = None

    # evaluation parameters
    test_n_z = 500
    test_batch_size = 128


config = ExpConfig()


@spt.global_reuse
@add_arg_scope
def q_net(x, posterior_flow, observed=None, n_z=None):
    net = spt.BayesianNet(observed=observed)

    # compute the hidden features
    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_x = tf.to_float(x)
        h_x = spt.layers.dense(h_x, 500)
        h_x = spt.layers.dense(h_x, 500)

    # sample z ~ q(z|x)
    z_mean = spt.layers.dense(h_x, config.z_dim, name='z_mean')
    z_logstd = spt.layers.dense(h_x, config.z_dim, name='z_logstd')
    z = net.add('z', spt.Normal(mean=z_mean, logstd=z_logstd), n_samples=n_z,
                group_ndims=1, flow=posterior_flow)

    return net


@spt.global_reuse
@add_arg_scope
def p_net(observed=None, n_z=None):
    net = spt.BayesianNet(observed=observed)

    # sample z ~ p(z)
    z = net.add('z', spt.Normal(mean=tf.zeros([1, config.z_dim]),
                                logstd=tf.zeros([1, config.z_dim])),
                group_ndims=1, n_samples=n_z)

    # compute the hidden features
    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_z = z
        h_z = spt.layers.dense(h_z, 500)
        h_z = spt.layers.dense(h_z, 500)

    # sample x ~ p(x|z)
    x_logits = spt.layers.dense(h_z, config.x_dim, name='x_logits')
    x = net.add('x', spt.Bernoulli(logits=x_logits), group_ndims=1)

    return net


def coupling_layer_shift_and_scale(x1, n2):
    # compute the hidden features
    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h = x1
        for _ in range(config.n_flow_hidden_layers):
            h = spt.layers.dense(h, 500)

    # compute shift and scale
    shift = spt.layers.dense(
        h, n2, kernel_initializer=tf.zeros_initializer(),
        bias_initializer=tf.zeros_initializer(), name='shift'
    )
    scale = spt.layers.dense(
        h, n2, kernel_initializer=tf.zeros_initializer(),
        bias_initializer=tf.zeros_initializer(), name='scale'
    )
    return shift, scale


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
    results.make_dirs('plotting', exist_ok=True)
    results.make_dirs('train_summary', exist_ok=True)

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.int32, shape=(None, config.x_dim), name='input_x')
    learning_rate = spt.AnnealingVariable(
        'learning_rate', config.initial_lr, config.lr_anneal_factor)

    # build the posterior flow
    with tf.variable_scope('posterior_flow'):
        flows = []
        for i in range(config.n_flows):
            flows.append(spt.layers.ActNorm())
            flows.append(spt.layers.CouplingLayer(
                tf.make_template(
                    'coupling',
                    coupling_layer_shift_and_scale,
                    create_scope_now_=True
                ),
                scale_type='exp'
            ))
            flows.append(spt.layers.InvertibleDense())
        posterior_flow = spt.layers.SequentialFlow(flows=flows)

    # derive the initialization op
    with tf.name_scope('initialization'), \
            arg_scope([spt.layers.act_norm], initializing=True):
        init_q_net = q_net(input_x, posterior_flow)
        init_chain = init_q_net.chain(
            p_net, latent_axis=0, observed={'x': input_x})
        init_loss = tf.reduce_mean(init_chain.vi.training.sgvb())

    # derive the loss and lower-bound for training
    with tf.name_scope('training'):
        train_q_net = q_net(input_x, posterior_flow)
        train_chain = train_q_net.chain(
            p_net, latent_axis=0, observed={'x': input_x})

        vae_loss = tf.reduce_mean(train_chain.vi.training.sgvb())
        loss = vae_loss + tf.losses.get_regularization_loss()

    # derive the nll and logits output for testing
    with tf.name_scope('testing'):
        test_q_net = q_net(input_x, posterior_flow, n_z=config.test_n_z)
        test_chain = test_q_net.chain(
            p_net, latent_axis=0, observed={'x': input_x})
        test_nll = -tf.reduce_mean(
            test_chain.vi.evaluation.is_loglikelihood())
        test_lb = tf.reduce_mean(test_chain.vi.lower_bound.elbo())

    # derive the optimizer
    with tf.name_scope('optimizing'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        params = tf.trainable_variables()
        grads = optimizer.compute_gradients(loss, var_list=params)
        with tf.control_dependencies(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.apply_gradients(grads)

    # derive the plotting function
    with tf.name_scope('plotting'):
        plot_p_net = p_net(n_z=100)
        x_plots = tf.reshape(bernoulli_as_pixel(plot_p_net['x']), (-1, 28, 28))

    def plot_samples(loop):
        with loop.timeit('plot_time'):
            images = session.run(x_plots)
            save_images_collection(
                images=images,
                filename='plotting/{}.png'.format(loop.epoch),
                grid_size=(10, 10)
            )

    # prepare for training and testing data
    (x_train, y_train), (x_test, y_test) = spt.datasets.load_mnist()
    train_flow = bernoulli_flow(
        x_train, config.batch_size, shuffle=True, skip_incomplete=True)
    test_flow = bernoulli_flow(
        x_test, config.test_batch_size, sample_now=True)

    with spt.utils.create_session().as_default() as session, \
            train_flow.threaded(5) as train_flow:
        # initialize the network
        spt.utils.ensure_variables_initialized()
        for [batch_x] in train_flow:
            print('Network initialization loss: {:.6g}'.
                  format(session.run(init_loss, {input_x: batch_x})))
            print('')
            break

        # train the network
        with spt.TrainLoop(params,
                           var_groups=['p_net', 'q_net', 'posterior_flow'],
                           max_epoch=config.max_epoch,
                           max_step=config.max_step,
                           summary_dir=(results.system_path('train_summary')
                                        if config.write_summary else None),
                           summary_graph=tf.get_default_graph(),
                           early_stopping=False) as loop:
            trainer = spt.Trainer(
                loop, train_op, [input_x], train_flow,
                metrics={'loss': loss}
            )
            trainer.anneal_after(
                learning_rate,
                epochs=config.lr_anneal_epoch_freq,
                steps=config.lr_anneal_step_freq
            )
            evaluator = spt.Evaluator(
                loop,
                metrics={'test_nll': test_nll, 'test_lb': test_lb},
                inputs=[input_x],
                data_flow=test_flow,
                time_metric_name='test_time'
            )
            evaluator.after_run.add_hook(
                lambda: results.update_metrics(evaluator.last_metrics_dict))
            trainer.evaluate_after_epochs(evaluator, freq=10)
            trainer.evaluate_after_epochs(
                functools.partial(plot_samples, loop), freq=10)
            trainer.log_after_epochs(freq=1)
            trainer.run()

    # print the final metrics and close the results object
    print_with_title('Results', results.format_metrics(), before='\n')
    results.close()


if __name__ == '__main__':
    main()
