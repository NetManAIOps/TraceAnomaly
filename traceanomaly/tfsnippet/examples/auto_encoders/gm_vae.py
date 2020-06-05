# -*- coding: utf-8 -*-
import codecs
import functools
import sys
import warnings
from argparse import ArgumentParser

import tensorflow as tf
from pprint import pformat
from sklearn.metrics import accuracy_score
from tensorflow.contrib.framework import arg_scope, add_arg_scope

import tfsnippet as spt
from tfsnippet.examples.utils import (MLResults,
                                      save_images_collection,
                                      collect_outputs,
                                      ClusteringClassifier,
                                      bernoulli_as_pixel,
                                      bernoulli_flow,
                                      print_with_title)


class ExpConfig(spt.Config):
    # model parameters
    x_dim = 784
    z_dim = 16
    n_clusters = 16
    l2_reg = 0.0001
    p_z_given_y_std = spt.ConfigField(
        str, default='unbound_logstd', choices=[
            'one', 'one_plus_softplus_std', 'softplus_logstd', 'unbound_logstd'
        ]
    )
    mean_field_assumption_for_q = False

    # training parameters
    result_dir = None
    write_summary = False
    max_epoch = 3000
    max_step = None
    batch_size = 128
    vi_algorithm = spt.ConfigField(
        str, default='vimco', choices=['reinforce', 'vimco'])
    train_n_samples = 25

    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 300
    lr_anneal_step_freq = None

    # evaluation parameters
    test_n_samples = 500
    test_batch_size = 128


config = ExpConfig()


@spt.global_reuse
def gaussian_mixture_prior(y, z_dim, n_clusters):
    # derive the learnt z_mean
    prior_mean = spt.model_variable(
        'z_prior_mean', dtype=tf.float32, shape=[n_clusters, z_dim],
        initializer=tf.random_normal_initializer()
    )
    z_mean = tf.nn.embedding_lookup(prior_mean, y)

    # derive the learnt z_std
    z_logstd = z_std = None
    if config.p_z_given_y_std == 'one':
        z_logstd = tf.zeros_like(z_mean)
    else:
        prior_std_or_logstd = spt.model_variable(
            'z_prior_std_or_logstd',
            dtype=tf.float32,
            shape=[n_clusters, z_dim],
            initializer=tf.zeros_initializer()
        )
        z_std_or_logstd = tf.nn.embedding_lookup(prior_std_or_logstd, y)

        if config.p_z_given_y_std == 'one_plus_softplus_std':
            z_std = 1. + tf.nn.softplus(z_std_or_logstd)
        elif config.p_z_given_y_std == 'softplus_logstd':
            z_logstd = tf.nn.softplus(z_std_or_logstd)
        else:
            assert(config.p_z_given_y_std == 'unbound_logstd')
            z_logstd = z_std_or_logstd

    return spt.Normal(mean=z_mean, std=z_std, logstd=z_logstd)


@spt.global_reuse
@add_arg_scope
def q_net(x, observed=None, n_samples=None):
    net = spt.BayesianNet(observed=observed)

    # compute the hidden features
    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_x = tf.to_float(x)
        h_x = spt.layers.dense(h_x, 500)
        h_x = spt.layers.dense(h_x, 500)

    # sample y ~ q(y|x)
    y_logits = spt.layers.dense(h_x, config.n_clusters, name='y_logits')
    y = net.add('y', spt.Categorical(y_logits), n_samples=n_samples)
    y_one_hot = tf.one_hot(y, config.n_clusters, dtype=tf.float32)

    # sample z ~ q(z|y,x)
    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        if config.mean_field_assumption_for_q:
            # by mean-field-assumption we let q(z|y,x) = q(z|x)
            h_z = h_x
            z_n_samples = n_samples
        else:
            if n_samples is not None:
                h_z = tf.concat(
                    [
                        tf.tile(tf.reshape(h_x, [1, -1, 500]),
                                tf.stack([n_samples, 1, 1])),
                        y_one_hot
                    ],
                    axis=-1
                )
            else:
                h_z = tf.concat([h_x, y_one_hot], axis=-1)
            h_z = spt.layers.dense(h_z, 500)
            z_n_samples = None

    z_mean = spt.layers.dense(h_z, config.z_dim, name='z_mean')
    z_logstd = spt.layers.dense(h_z, config.z_dim, name='z_logstd')
    z = net.add('z',
                spt.Normal(mean=z_mean, logstd=z_logstd,
                           is_reparameterized=False),
                n_samples=z_n_samples, group_ndims=1)

    return net


@spt.global_reuse
@add_arg_scope
def p_net(observed=None, n_y=None, n_z=None, n_samples=None):
    if n_samples is not None:
        warnings.warn('`n_samples` is deprecated, use `n_y` instead.')
        n_y = n_samples

    net = spt.BayesianNet(observed=observed)

    # sample y
    y = net.add('y',
                spt.Categorical(tf.zeros([1, config.n_clusters])),
                n_samples=n_y)

    # sample z ~ p(z|y)
    z = net.add('z',
                gaussian_mixture_prior(y, config.z_dim, config.n_clusters),
                group_ndims=1,
                n_samples=n_z,
                is_reparameterized=False)

    # compute the hidden features for x
    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_x = z
        h_x = spt.layers.dense(h_x, 500)
        h_x = spt.layers.dense(h_x, 500)

    # sample x ~ p(x|z)
    x_logits = spt.layers.dense(h_x, config.x_dim, name='x_logits')
    x = net.add('x', spt.Bernoulli(logits=x_logits), group_ndims=1)

    return net


@spt.global_reuse
def reinforce_baseline_net(x):
    with arg_scope([spt.layers.dense],
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg),
                   activation_fn=tf.nn.leaky_relu):
        h_x = tf.to_float(x)
        h_x = spt.layers.dense(h_x, 500)
    h_x = tf.squeeze(spt.layers.dense(h_x, 1), axis=-1)
    return h_x


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

    # derive the loss and lower-bound for training
    with tf.name_scope('training'):
        train_q_net = q_net(
            input_x, n_samples=config.train_n_samples
        )
        train_chain = train_q_net.chain(
            p_net, latent_axis=0, observed={'x': input_x})

        if config.vi_algorithm == 'reinforce':
            baseline = reinforce_baseline_net(input_x)
            vae_loss = tf.reduce_mean(
                train_chain.vi.training.reinforce(baseline=baseline))
        else:
            assert(config.vi_algorithm == 'vimco')
            vae_loss = tf.reduce_mean(train_chain.vi.training.vimco())
        loss = vae_loss + tf.losses.get_regularization_loss()

    # derive the nll and logits output for testing
    with tf.name_scope('testing'):
        test_q_net = q_net(
            input_x, n_samples=config.test_n_samples
        )
        test_chain = test_q_net.chain(
            p_net, latent_axis=0, observed={'x': input_x})
        test_nll = -tf.reduce_mean(
            test_chain.vi.evaluation.is_loglikelihood())

        # derive the classifier via q(y|x)
        q_y_given_x = tf.argmax(test_q_net['y'].distribution.logits,
                                axis=-1, name='q_y_given_x')

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
        plot_p_net = p_net(
            observed={'y': tf.range(config.n_clusters, dtype=tf.int32)},
            n_z=10
        )
        x_plots = tf.reshape(
            tf.transpose(bernoulli_as_pixel(plot_p_net['x']), (1, 0, 2)),
            (-1, 28, 28)
        )

    def plot_samples(loop):
        with loop.timeit('plot_time'):
            images = session.run(x_plots)
            save_images_collection(
                images=images,
                filename='plotting/{}.png'.format(loop.epoch),
                grid_size=(config.n_clusters, 10),
                results=results
            )

    # derive the final un-supervised classifier
    c_classifier = ClusteringClassifier(config.n_clusters, 10)

    def train_classifier(loop):
        df = bernoulli_flow(
            x_train, config.batch_size, shuffle=False, skip_incomplete=False)
        with loop.timeit('cls_train_time'):
            [c_pred] = collect_outputs(
                outputs=[q_y_given_x],
                inputs=[input_x],
                data_flow=df,
            )
            c_classifier.fit(c_pred, y_train)
            print(c_classifier.describe())

    def evaluate_classifier(loop):
        with loop.timeit('cls_test_time'):
            [c_pred] = collect_outputs(
                outputs=[q_y_given_x],
                inputs=[input_x],
                data_flow=test_flow,
            )
            y_pred = c_classifier.predict(c_pred)
            cls_metrics = {'test_acc': accuracy_score(y_test, y_pred)}
            loop.collect_metrics(cls_metrics)
            results.update_metrics(cls_metrics)

    # prepare for training and testing data
    (x_train, y_train), (x_test, y_test) = spt.datasets.load_mnist()
    train_flow = bernoulli_flow(
        x_train, config.batch_size, shuffle=True, skip_incomplete=True)
    test_flow = bernoulli_flow(
        x_test, config.test_batch_size, sample_now=True)

    with spt.utils.create_session().as_default() as session, \
            train_flow.threaded(5) as train_flow:
        # train the network
        with spt.TrainLoop(params,
                           var_groups=['p_net', 'q_net',
                                       'gaussian_mixture_prior'],
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
                metrics={'test_nll': test_nll},
                inputs=[input_x],
                data_flow=test_flow,
                time_metric_name='test_time'
            )
            evaluator.after_run.add_hook(
                lambda: results.update_metrics(evaluator.last_metrics_dict))
            trainer.evaluate_after_epochs(evaluator, freq=10)
            trainer.evaluate_after_epochs(
                functools.partial(plot_samples, loop), freq=10)
            trainer.evaluate_after_epochs(
                functools.partial(train_classifier, loop), freq=10)
            trainer.evaluate_after_epochs(
                functools.partial(evaluate_classifier, loop), freq=10)

            trainer.log_after_epochs(freq=1)
            trainer.run()

    # print the final metrics and close the results object
    with codecs.open('cluster_classifier.txt', 'wb', 'utf-8') as f:
        f.write(c_classifier.describe())
    print_with_title('Results', results.format_metrics(), before='\n')
    results.close()


if __name__ == '__main__':
    main()
