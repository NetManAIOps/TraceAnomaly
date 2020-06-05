import multiprocessing as mp
import traceback
from contextlib import contextmanager

import six
import tensorflow as tf

from tfsnippet.utils import (is_tensor_object,
                             is_tensorflow_version_higher_or_equal)
from .misc import cached

__all__ = ['detect_gpus', 'average_gradients', 'MultiGPU']


@cached
def detect_gpus():
    """
    Detect the GPU devices and their interconnection on current machine.

    Returns:
        list[list[str]]: List of GPU groups, each group is a list of
            GPU device names.  The GPUs in one group are interconnected.
    """
    def worker(q):
        # `device_lib` will not release the memory it took,
        # so we run it in a sub-process.
        try:
            from tensorflow.python.client import device_lib

            if is_tensorflow_version_higher_or_equal('1.8.0'):
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                devices = list(device_lib.list_local_devices(config))
            else:
                devices = list(device_lib.list_local_devices())
            gpus = [
                (device.name, device)
                for device in devices
                if device.device_type == 'GPU'
            ]
            union_set = {i: i for i in range(len(gpus))}

            for i, (name, device) in enumerate(gpus):
                assert (device.name == '/device:GPU:{}'.format(i))
                for link in device.locality.links.link:
                    if link.device_id != i:
                        union_set[i] = union_set[link.device_id]

            for i in six.iterkeys(union_set):
                while union_set[i] != union_set[union_set[i]]:
                    union_set[i] = union_set[union_set[i]]

            root_devices = sorted(set(union_set.values()))
            gpu_groups = [[] for _ in range(len(root_devices))]
            dev_to_group = {j: i for i, j in enumerate(root_devices)}
            for i, (name, device) in enumerate(gpus):
                gpu_groups[dev_to_group[union_set[i]]].append(name)

            q.put((1, gpu_groups))
        except Exception:
            q.put((0, traceback.format_exc()))

    q = mp.Queue()
    p = mp.Process(target=worker, args=(q,))

    try:
        p.start()
        result = q.get()
        if result[0] == 1:
            return result[1]
        else:
            raise RuntimeError(
                'Failed to retrieve GPU information, the traceback of '
                'sub-process is:\n  {}'.
                format('\n  '.join(result[1].split('\n')))
            )
    finally:
        p.terminate()
        p.join()


def average_gradients(tower_grads):
    """
    Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.

    Source:
        https://github.com/tensorflow/models/blob/master/tutorials/image/
        cifar10/cifar10_multi_gpu_train.py

    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer
            list is over individual gradients. The inner list is over the
            gradient calculation for each tower.

    Returns:
       List of pairs of (gradient, variable) where the gradient has been
       averaged across all towers.
    """
    if len(tower_grads) == 1:
        return tower_grads[0]

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


class MultiGPU(object):
    """
    Class to help build data-paralleled outputs and training operations.
    """

    def __init__(self, disable_prebuild=False):
        """
        Construct a :class:`MultiGPU`.

        Args:
            disable_prebuild: Whether or not to disable pre-build on CPU?
                Some operations (e.g., NCHW convolutional kernels) may not be
                supported by CPUs for the time being, thus the pre-building on
                CPUs might need to be disabled.
        """
        gpu_groups = detect_gpus()
        if not gpu_groups:
            self._main_device = '/device:CPU:0'
        elif len(gpu_groups) != 1 and not disable_prebuild:
            self._main_device = '/device:CPU:0'
        else:
            self._main_device = gpu_groups[0][0]

        self._disable_prebuild = disable_prebuild
        self._gpu_devices = tuple(sum(gpu_groups, []))
        self._work_devices = self._gpu_devices \
            if self._gpu_devices else [self._main_device]

    @property
    def disable_prebuild(self):
        """Whether or not to disable pre-build on CPU?"""
        return self._disable_prebuild

    @property
    def main_device(self):
        """
        Get the main device name.

        Main device is the device for storing variables, and for gathering
        losses / gradients during training.  It may not be necessary one
        of the `work_devices`.  Do not run the model computation graph on the
        `main_device`, otherwise the `channels_last` parameter for convolutional
        layers might result in undesired behaviors.
        """
        return self._main_device

    @property
    def work_devices(self):
        """
        Get the names of the working devices.

        The model computation graph should be run only on these devices.
        Do not run them on the `main_device`, otherwise the `channels_last`
        parameter for convolutional layers might result in undesired behaviors.
        """
        return self._work_devices

    @property
    def gpu_devices(self):
        """Get the names of GPU devices."""
        return self._gpu_devices

    def is_gpu_device(self, device):
        """Check whether or not `device` is a GPU device."""
        return device in self._gpu_devices

    def channels_last(self, device):
        """
        Get the `channels_last` argument for `device`.

        It will be :obj:`True` for non-GPU devices, :obj:`False` for GPUs.
        Be careful if you want to build a model on both CPU and GPU devices,
        with ``channels_last = multi_gpu.channels_last(device)``.
        The convolutional layers will work as desired, but the dense layers
        after or before a convolutional layer will not work properly, unless
        special treatment is taken.
        """
        return device not in self._gpu_devices

    def data_parallel(self, batch_size, inputs):
        """
        Iterate through all devices and build the data-paralleled model.

        Args:
            batch_size (int or tf.Tensor): The size of each mini-batch.
            inputs (Iterable[tf.Tensor]): Input placeholders to be sliced
                for data parallelism.  The input placeholders will be sliced
                through the first dimension.

        Yields:
            str, bool, tuple[tf.Tensor]: ``(dev, pre_build, inputs)``,
                the device name, a flag indicating whether this is a
                pre-building pass for creating variables on CPU, and the
                tuple of sliced input placeholders.
        """
        inputs = list(inputs)

        # quick path: only one device, do not slice
        if len(self.work_devices) == 1:
            assert(self.main_device == self.work_devices[0])
            yield self.main_device, False, tuple(inputs)

        # slow path: multi-GPUs
        else:
            # the GPUs are not in the same group, place variables on CPU
            if self.main_device not in self.work_devices:
                yield self.main_device, True, tuple(inputs)

            # build the paralleled computation graph for each device
            with tf.name_scope('data_parallel') as ns:
                pass  # generate a name scope to place our data slicing ops

            k = len(self.work_devices)
            for i, device in enumerate(self.work_devices):
                dev_inputs = []
                with tf.name_scope(ns + 'tower_gpu_{}'.format(i)):
                    for inp in inputs:
                        slice_len = (batch_size + k - 1) // k
                        low, high = slice_len * i, slice_len * (i + 1)
                        dev_inputs.append(inp[low: high])
                yield device, False, tuple(dev_inputs)

    @contextmanager
    def maybe_name_scope(self, device):
        """
        Generate a name scope if `device` is not `main_device`.

        Args:
            device (str): The name of the device.

        Yields
            The generated name scope, or None.
        """
        if device == self.main_device:
            yield
        elif device not in self._gpu_devices:
            with tf.name_scope('tower_cpu') as ns:
                yield ns
        else:
            gpu_id = self._gpu_devices.index(device)
            with tf.name_scope('tower_gpu_{}'.format(gpu_id)) as ns:
                yield ns

    def average_grads(self, grads):
        """
        Take the averaged gradients on the main device.

        Args:
            grads: List of lists of (gradients, variables) pairs.

        Returns:
            List of pairs of (gradient, variable) where the gradient has been
            averaged across all devices.
        """
        # quick path: only one device, just return the grads
        if len(grads) == 1:
            return grads[0]

        # slow path: multi-GPUs
        else:
            with tf.device(self.main_device), tf.name_scope('average_grads'):
                return average_gradients(grads)

    def apply_grads(self, grads, optimizer, global_step=None,
                    control_inputs=None):
        """
        Apply the gradients.

        Args:
            grads: List of (gradients, variables) pairs.
            optimizer: The TensorFlow optimizer.
            global_step: The optional global step counter.
            control_inputs: Dependency operations before applying the gradients.

        Returns:
            The operation of applying gradients.
        """
        def mk_op():
            return optimizer.apply_gradients(grads, global_step=global_step)

        with tf.device(self.main_device), tf.name_scope('apply_grads'):
            if control_inputs:
                with tf.control_dependencies(control_inputs):
                    return mk_op()
            else:
                return mk_op()

    def average(self, tensors, batch_size=None):
        """
        Take the average of given tensors from different devices.

        If `batch_size` is specified, the tensors will be averaged with respect
        to the size of data fed to each device.

        Args:
            tensors (list[list[tf.Tensor]]): List of tensors from each device.
            batch_size (None or int or tf.Tensor): The optional batch size.

        Returns:
            list[tf.Tensor]: The averaged tensors.
        """
        # check the arguments and try the fast path: only one tensor
        tensors = list(tensors)
        if not tensors:
            return []
        length = len(tensors[0])
        if length == 0:
            raise ValueError('`tensors` must be list of non-empty Tensor '
                             'lists.')
        for t in tensors[1:]:
            if len(t) != length:
                raise ValueError('`tensors` must be list of Tensor lists of '
                                 'the same length.')
        if length == 1:
            return [t[0] for t in tensors]

        # do the slow path: average all tensors
        with tf.device(self.main_device), tf.name_scope('average_tensors'):
            if batch_size is None:
                return [tf.reduce_mean(tf.stack(t), axis=0) for t in tensors]

            k = len(self.work_devices)
            slice_len = (batch_size + k - 1) // k
            last_slice_size = batch_size - (k - 1) * slice_len

            if is_tensor_object(batch_size):
                to_float = tf.to_float
            else:
                to_float = float

            float_batch_size = to_float(batch_size)
            weights = tf.stack(
                [to_float(slice_len) / float_batch_size] * (k - 1) +
                [to_float(last_slice_size) / float_batch_size]
            )

            return [tf.reduce_sum(tf.stack(t) * weights, axis=0)
                    for t in tensors]

    def concat(self, tensors):
        """
        Concat given tensors from different devices.

        Args:
            tensors (list[list[tf.Tensor]]): List of tensors from each device.

        Returns:
            list[tf.Tensor]: The concatenated tensors.
        """
        # check the arguments and try the fast path: only one tensor
        tensors = list(tensors)
        if not tensors:
            return []
        length = len(tensors[0])
        if length == 0:
            raise ValueError('`tensors` must be list of non-empty Tensor '
                             'lists.')
        for t in tensors[1:]:
            if len(t) != length:
                raise ValueError('`tensors` must be list of Tensor lists of '
                                 'the same length.')
        if length == 1:
            return [t[0] for t in tensors]

        # do the slow path: concat all tensors
        with tf.device(self.main_device), tf.name_scope('average_tensors'):
            return [tf.concat(t, axis=0) for t in tensors]
