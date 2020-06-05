# -*- coding: utf-8 -*-
import functools

import os, time
import click
import tensorflow as tf
import pandas as pd
from tensorflow.contrib.framework import arg_scope, add_arg_scope
from sklearn.model_selection import train_test_split

import tfsnippet as spt
from tfsnippet.examples.utils import (print_with_title,
                                      collect_outputs)
from .readdata import get_data_vae, get_z_dim
from .MLConfig import (MLConfig,
                       global_config as config,
                       config_options)

class ExpConfig(MLConfig):
    debug_level = -1  # -1: disable all checks;
                      #  0: assertions only
                      #  1: assertions + numeric check


    # model parameters
    z_dim = 10
    x_dim = 100
    
    flow_type = None # None: no flow
                       # planar_nf:
                       # rnvp
    n_planar_nf_layers = 10
    n_rnvp_layers = 10
    n_rnvp_hidden_layers = 1

    # training parameters
    write_summary = False
    max_epoch = 2000
    max_step = None
    batch_size = 256
    
    l2_reg = 0.0001
    initial_lr = 0.001
    #l2_reg = 0.001
    #initial_lr = 0.01
    
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 100
    lr_anneal_step_freq = None

    # evaluation parameters
    test_n_z = 500
    test_batch_size = 128

    norm_clip = 10


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
    z_std = 1e-4 + tf.nn.softplus(
        spt.layers.dense(h_x, config.z_dim, name='z_std'))
    z = net.add('z', spt.Normal(mean=z_mean, std=z_std), n_samples=n_z,
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
    x_mean = spt.layers.dense(h_z, config.x_dim, name='x_mean')
    x_std = 1e-4 + tf.nn.softplus(
        spt.layers.dense(h_z, config.x_dim, name='x_std'))
    x = net.add('x', spt.Normal(mean=x_mean, std=x_std), group_ndims=1)

    return net


def coupling_layer_shift_and_scale(x1, n2):
    # compute the hidden features
    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h = x1
        for _ in range(config.n_rnvp_hidden_layers):
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


@click.command()
@click.option('--trainpath', help='The path of train data', metavar='PATH',
              required=True, type=str)
@click.option('--normalpath', help='The path of normal data', metavar='PATH',
              required=True, type=str)
@click.option('--abnormalpath', help='The path of normal data', metavar='PATH',
              required=True, type=str)
@click.option('--outputpath',
              help='The name of answers. it is relative to webankdata. just name',
              metavar='PATH',
              required=True, type=str)
@config_options(ExpConfig)
def main(trainpath, normalpath, abnormalpath, outputpath):
    if config.debug_level == -1:
        spt.utils.set_assertion_enabled(False)
    elif config.debug_level == 1:
        spt.utils.set_check_numerics(True)

    #spt.utils.set_assertion_enabled(False)
    # print the config
    print_with_title('Configurations', config.format_config(), after='\n')

    # input and output file
    train_file = trainpath
    normal_file = normalpath
    abnormal_file = abnormalpath
    output_file = os.path.join('webankdata',
                               '{}_{}.csv'.format(config.flow_type or 'vae',
                                                  outputpath))
    valid_file = os.path.join('webankdata',
                              'v{}_{}.csv'.format(config.flow_type or 'vae',
                                                  outputpath))
    # you can change it by yourself

    # read data
    (x_train, y_train), (x_test, y_test), flows_test = \
        get_data_vae(train_file, normal_file, abnormal_file)
    config.x_dim = x_train.shape[1]
    #config.z_dim = get_z_dim(x_train.shape[1])

    all_len = x_train.shape[0]
    print('origin data: %s' % all_len)
    for i in range(30):
        print(list(x_train[i]))
    
    valid_rate = 0.1
    x_train, x_valid = train_test_split(x_train, test_size=valid_rate)
    
    
    # x_valid = x_train
    print('%s for validation, %s for training v2' % (x_valid.shape[0], x_train.shape[0]))
    print('%s for test' % x_test.shape[0])

    print('x_dim: %s z_dim: %s' % (config.x_dim, config.z_dim))
    # change it by yourself

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.float32, shape=(None, config.x_dim), name='input_x')
    learning_rate = spt.AnnealingVariable(
        'learning_rate', config.initial_lr, config.lr_anneal_factor)

    # build the posterior flow
    if config.flow_type is None:
        posterior_flow = None
    elif config.flow_type == 'planar_nf':
        posterior_flow = \
            spt.layers.planar_normalizing_flows(config.n_planar_nf_layers)
    else:
        assert(config.flow_type == 'rnvp')
        with tf.variable_scope('posterior_flow'):
            flows = []
            for i in range(config.n_rnvp_layers):
                flows.append(spt.layers.ActNorm())
                flows.append(spt.layers.CouplingLayer(
                    tf.make_template(
                        'coupling',
                        coupling_layer_shift_and_scale,
                        create_scope_now_=True
                    ),
                    scale_type='sigmoid'
                ))
                flows.append(spt.layers.InvertibleDense(strict_invertible=True))
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
        test_logp = test_chain.vi.evaluation.is_loglikelihood()
        test_nll = -tf.reduce_mean(test_logp)
        test_lb = tf.reduce_mean(test_chain.vi.lower_bound.elbo())

    # derive the optimizer
    with tf.name_scope('optimizing'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        params = tf.trainable_variables()
        grads = optimizer.compute_gradients(loss, var_list=params)

        cliped_grad = []
        for grad, var in grads:
            if grad is not None and var is not None:
                if config.norm_clip is not None:
                    grad = tf.clip_by_norm(grad, config.norm_clip)
                cliped_grad.append((grad, var))

        with tf.control_dependencies(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.apply_gradients(cliped_grad)

    train_flow = spt.DataFlow.arrays([x_train],
                                     config.batch_size,
                                     shuffle=True,
                                     skip_incomplete=True)
    valid_flow = spt.DataFlow.arrays([x_valid],
                                     config.test_batch_size)
    test_flow = spt.DataFlow.arrays([x_test],
                                    config.test_batch_size)

    # model_file
    #model_name = ''
    model_name = os.path.join(
        'webankdata',
        'md_{}_{}.model'.format(
            config.flow_type or 'vae',
            outputpath.split('.')[0]
        )
    )

    with spt.utils.create_session().as_default() as session:
        var_dict = spt.utils.get_variables_as_dict()
        saver = spt.VariableSaver(var_dict, model_name)
        #if os.path.exists(model_name):
        if False:
            print('%s exists' % model_name)
            saver.restore()
        else:
            print('no model here, and start training')
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
                               early_stopping=True,
                               valid_metric_name='valid_loss',
                               valid_metric_smaller_is_better=True) as loop:
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
                    metrics={'valid_loss': test_nll},
                    inputs=[input_x],
                    data_flow=valid_flow,
                    time_metric_name='valid_time'
                )

                trainer.evaluate_after_epochs(evaluator, freq=10)
                trainer.log_after_epochs(freq=1)
                trainer.run()
            saver.save()

        # get the answer
        print('start testing')
        start = time.time()
        test_ans = collect_outputs([test_logp], [input_x], test_flow)[0] \
            / config.x_dim
        end = time.time()
        print("test time: ", end-start)
        
        pd.DataFrame(
            {'id': flows_test, 'label': y_test, 'score': test_ans}) \
            .to_csv(output_file, index=False)
        valid_ans = collect_outputs([test_logp], [input_x], valid_flow)[0] \
            / config.x_dim
        pd.DataFrame({'score': valid_ans}).to_csv(valid_file, index=False)


if __name__ == '__main__':
    main()