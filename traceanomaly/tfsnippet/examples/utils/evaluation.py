import imageio
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from tfsnippet.distributions import Bernoulli
from tfsnippet.stochastic import StochasticTensor
from tfsnippet.trainer import merge_feed_dict, resolve_feed_dict
from tfsnippet.utils import get_default_session_or_error
from .mlresults import MLResults

__all__ = [
    'collect_outputs',  'save_images_collection', 'plot_2d_log_p',
    'ClusteringClassifier', 'bernoulli_as_pixel'
]


def collect_outputs(outputs, inputs, data_flow, feed_dict=None, session=None):
    """
    Run TensorFlow graph by mini-batch and concat outputs from each batch.

    Args:
        outputs (Iterable[tf.Tensor]): Output tensors to be computed.
        inputs (Iterable[tf.Tensor]): Input placeholders.
        data_flow (DataFlow): Data flow to feed the input placeholders.
        feed_dict: Optional, additional feed dict.
        session: The TensorFlow session.  If not specified, use the
            default session.

    Returns:
        tuple[np.ndarray]: The concatenated outputs.
    """
    outputs = list(outputs)
    inputs = list(inputs)
    session = session or get_default_session_or_error()

    collected = [[] for _ in range(len(outputs))]
    for batch in data_flow:
        batch_feed_dict = merge_feed_dict(
            feed_dict,
            {k: v for (k, v) in zip(inputs, batch)}
        )
        batch_feed_dict = resolve_feed_dict(batch_feed_dict)
        for i, o in enumerate(session.run(outputs, feed_dict=batch_feed_dict)):
            collected[i].append(o)

    for i, batches in enumerate(collected):
        collected[i] = np.concatenate(batches, axis=0)
    return tuple(collected)


def save_images_collection(images, filename, grid_size, border_size=0,
                           channels_last=True, results=None):
    """
    Save a collection of images as a large image, arranged in grid.

    Args:
        images: The images collection.  Each element should be a Numpy array,
            in the shape of ``(H, W)``, ``(H, W, C)`` (if `channels_last` is
            :obj:`True`) or ``(C, H, W)``.
        filename (str): The target filename.
        grid_size ((int, int)): The ``(rows, columns)`` of the grid.
        border_size (int): Size of the border, for separating images.
            (default 0, no border)
        channels_last (bool): Whether or not the channel dimension is at last?
        results (MLResults): If specified, will save the image via this
            :class:`Results` instance.  If not specified, will save the image
            to `filename` on local file system.
    """
    # check the arguments
    def validate_image(img):
        if len(img.shape) == 2:
            img = np.reshape(img, img.shape + (1,))
        elif len(images[0].shape) == 3:
            if img.shape[2 if channels_last else 0] not in (1, 3, 4):
                raise ValueError('Unexpected image shape: {!r}'.
                                 format(img.shape))
            if not channels_last:
                img = np.transpose(img, (1, 2, 0))
        else:
            raise ValueError('Unexpected image shape: {!r}'.format(img.shape))
        return img

    images = [validate_image(img) for img in images]
    h, w = images[0].shape[:2]
    rows, cols = grid_size[0], grid_size[1]
    buf_h = rows * h + (rows - 1) * border_size
    buf_w = cols * w + (cols - 1) * border_size

    # copy the images to canvas
    n_channels = images[0].shape[2]
    buf = np.zeros((buf_h, buf_w, n_channels), dtype=images[0].dtype)
    for j in range(rows):
        for i in range(cols):
            img = images[j * cols + i]
            buf[j * (h + border_size): (j + 1) * h + j * border_size,
                i * (w + border_size): (i + 1) * w + i * border_size,
                :] = img[:, :, :]

    # save the image
    if n_channels == 1:
        buf = np.reshape(buf, (buf_h, buf_w))

    if results is not None:
        results.save_image(filename, buf)
    else:
        imageio.imwrite(filename, buf)


def plot_2d_log_p(x, log_p, cmap='jet', **kwargs):
    """
    Plot :math:`log p(x)` for 2-d `x`.

    Args:
        x: 3-d Tensor of shape (?, ?, 2).
        log_p: 2-d Tensor of shape (?, ?), i.e., x.shape[:2].
        cmap: The color map for plotting :math:`log p(x)`. (default "jet")
        \\**kwargs: Additional named arguments passed to ``plt.figure``.

    Returns:
        plt.Figure: The plotted figure.
    """
    x = np.asarray(x)
    if not len(x.shape) == 3 or x.shape[2] != 2:
        raise ValueError('The shape of `x` must be (?, ?, 2), got {!r}'.
                         format(x.shape))
    log_p = np.asarray(log_p)
    if log_p.shape != x.shape[:2]:
        raise ValueError('The shape of `log_p` must be x.shape[:2], got {!r}'.
                         format(log_p.shape))

    fig = plt.figure(**kwargs)
    cmap = plt.get_cmap(cmap)
    z = np.exp(log_p)
    h = plt.pcolormesh(x[:, :, 0], x[:, :, 1], z[:-1, :-1], cmap=cmap)
    plt.colorbar(h)
    return fig


class ClusteringClassifier(object):
    """
    Un-supervised classifier based on clustering algorithm.

    The performance of a clustering algorithm can be evaluated by the
    proxy of its classification performance, once given true class labels.
    """

    def __init__(self, n_clusters, n_classes):
        """
        Construct a new :class:`ClusteringClassifier`.

        Args:
            n_clusters (int): Number of clusters.
            n_classes (int): Number of classes.
        """
        self.n_clusters = n_clusters
        self.n_classes = n_classes
        self.cluster_probs = np.zeros([n_clusters])
        self.cluster_class_probs = np.zeros([n_clusters, n_classes])
        self.cluster_classes = np.ones([n_clusters], dtype=np.int32) * -1
        self._A = np.identity(n_clusters, dtype=np.float32)
        self._B = np.identity(n_clusters * n_classes, dtype=np.float32)

    def describe(self):
        """
        Describe the clustering classifier.

        Returns:
            str: Description of the classifier.
        """
        ret = [
            'Cluster probs: [{}]'.format(
                ', '.join('{:.4g}'.format(p) for p in self.cluster_probs)),
            'Cluster labels: {}'.format(self.cluster_classes.tolist()),
            'Cluster label probs:'
        ]
        for i, label_prob in enumerate(self.cluster_class_probs):
            ret.append('  {}: [{}]'.format(
                i, ', '.join('{:.4g}'.format(p) for p in label_prob)))
        return '\n'.join(ret)

    def fit(self, c_pred, y_true):
        """
        Fit the clustering based classifier.

        Args:
            c_pred (np.ndarray): 1-d array, the predicted cluster indices.
            y_true (np.ndarray): 1-d array, the true class labels.
        """
        c_pred = np.asarray(c_pred)
        y_true = np.asarray(y_true)
        if len(c_pred.shape) != 1:
            raise ValueError('`c_pred` must be 1-d array.')
        if y_true.shape != c_pred.shape:
            raise ValueError('The shape of `y_true` must be equal to '
                             'that of `c_pred`.')
        self.cluster_probs = np.mean(self._A[c_pred], axis=0)
        class_probs = np.sum(self._B[c_pred * self.n_classes + y_true], axis=0)
        class_probs = class_probs.reshape([self.n_clusters, self.n_classes])
        class_probs = class_probs / np.maximum(
            np.sum(class_probs, axis=-1, keepdims=True), 1)
        self.cluster_class_probs = class_probs
        self.cluster_classes = np.argmax(class_probs, axis=-1)

    def predict(self, c_pred):
        """
        Predict the most likely label.

        Args:
            c_pred (np.ndarray): 1-d array, the predicted cluster indices.

        Returns:
            np.ndarray: 1-d array, the predicted class labels.
        """
        c_pred = np.asarray(c_pred)
        if len(c_pred.shape) != 1:
            raise ValueError('`c_pred` must be 1-d array.')
        return self.cluster_classes[c_pred]


def bernoulli_as_pixel(x=None, uint8=True, name=None):
    """
    Translate a Bernoulli random variable as pixel values.

    This function will use the probability of the Bernoulli random variable
    to take 1 as the pixel value.

    Args:
        x (StochasticTensor or Bernoulli or Tensor): It should be a
            :class:`StochasticTensor`, a :class:`Bernoulli` distribution,
            or a Tensor indicating the logits of the Bernoulli variable.
            If it is a :class:`StochasticTensor`, its distribution must
            be :class:`Bernoulli`.  The logits of the Bernoulli distribution
            will be used to compute the probability.
        uint8 (bool): Whether or not to convert the pixel value into uint8?
            If :obj:`True`, will multiple the Bernoulli probability by 255,
            then convert the dtype into tf.uint8.  Otherwise will use the
            probability (range in ``[0, 1]``) as the pixel value.
        name: TensorFlow name scope of the graph nodes.

    Returns:
        tf.Tensor: The translated pixel values.
    """
    if isinstance(x, StochasticTensor):
        assert(isinstance(x.distribution, Bernoulli))
        logits = x.distribution.logits
    elif isinstance(x, Bernoulli):
        logits = x.logits
    else:
        logits = tf.convert_to_tensor(x)

    with tf.name_scope(name, default_name='bernoulli_as_pixel',
                       values=[logits]):
        pixels = tf.sigmoid(logits)
        if uint8:
            pixels = tf.cast(255 * pixels, dtype=tf.uint8)

    return pixels
