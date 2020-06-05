import numpy as np
import tensorflow as tf
from scipy import linalg as la

from .debugging import maybe_check_numerics
from .doc_utils import add_name_arg_doc, add_name_and_scope_arg_doc
from .model_vars import model_variable
from .random import VarScopeRandomState
from .reuse import VarScopeObject
from .scope import reopen_variable_scope
from .tensor_spec import InputSpec
from .tfver import is_tensorflow_version_higher_or_equal
from .type_utils import is_integer

__all__ = ['PermutationMatrix', 'InvertibleMatrix']


class PermutationMatrix(object):
    """A non-trainable permutation matrix."""

    def __init__(self, data):
        """
        Construct a new :class:`PermutationMatrix`.

        Args:
            data (np.ndarray or Iterable[int]): A 2-d permutation matrix,
                or a list of integers as the row permutation indices.
        """
        def validate_data(data):
            if isinstance(data, np.ndarray) and len(data.shape) == 2:
                try:
                    epsilon = np.finfo(data.dtype).eps
                except ValueError:
                    epsilon = 0

                # check whether or not `data` is a permutation matrix
                if data.shape[0] != data.shape[1] or data.shape[0] < 1:
                    raise ValueError()

                for axis in (0, 1):
                    axis_sum = np.sum(data, axis=axis)
                    if not np.all(np.abs(1 - axis_sum) <= epsilon):
                        raise ValueError()
                    axis_max = np.max(data, axis=axis)
                    if not np.all(np.abs(1 - axis_max) <= epsilon):
                        raise ValueError()

                # compute the row & col permutation indices
                row_perm = np.argmax(data, axis=1).astype(np.int32)
                col_perm = np.argmax(data, axis=0).astype(np.int32)
            else:
                # check whether or not `data` is row permutation indices
                data = np.asarray(data, dtype=np.int32)
                if len(data.shape) != 1 or len(data) < 1:
                    raise ValueError()
                if np.max(data) != len(data) - 1 or np.min(data) != 0 or \
                        len(np.unique(data)) != len(data):
                    raise ValueError()

                # compute the row & col permutation indices
                row_perm = data
                col_perm = [0] * len(data)
                for i, j in enumerate(row_perm):
                    col_perm[j] = i
                col_perm = np.asarray(col_perm, dtype=np.int32)

            return tuple(row_perm), tuple(col_perm)

        try:
            self._row_perm, self._col_perm = validate_data(data)
        except ValueError:
            raise ValueError('`data` is not a valid permutation matrix or '
                             'row permutation indices: {!r}'.format(data))
        self._shape = (len(self._row_perm),) * 2

        # compute the determinant
        det = 1
        for i in range(len(self._row_perm) - 1):
            for j in range(i+1, len(self._row_perm)):
                if self._row_perm[i] > self._row_perm[j]:
                    det = -det
        self._det = float(det)

    def __repr__(self):
        return 'PermutationMatrix({!r})'.format(self._row_perm)

    @property
    def shape(self):
        """
        Get the shape of this permutation matrix.

        Returns:
            (int, int): The shape of this permutation matrix.
        """
        return self._shape

    def det(self):
        """
        Get the determinant of this permutation matrix.

        Returns:
            float: The determinant of this permutation matrix.
        """
        return self._det

    @property
    def row_permutation(self):
        """
        Get the row permutation indices.

        Returns:
            tuple[int]: The row permutation indices.
        """
        return self._row_perm

    @property
    def col_permutation(self):
        """
        Get the column permutation indices.

        Returns:
            tuple[int]: The column permutation indices.
        """
        return self._col_perm

    def get_numpy_matrix(self, dtype=np.int32):
        """
        Get the numpy permutation matrix.

        Args:
            dtype: The data type of the returned matrix.

        Returns:
            np.ndarray: A 2-d numpy matrix.
        """
        m = np.zeros(self.shape, dtype=dtype)
        m[range(self.shape[0]), self._row_perm] = 1
        return m

    @add_name_arg_doc
    def left_mult(self, input, name=None):
        """
        Left multiply to `input` matrix.

        `output = matmul(self, input)`

        Args:
            input (np.ndarray or tf.Tensor): The input matrix, whose
                shape must be ``(self.shape[1], ?)``.

        Returns:
            np.ndarray or tf.Tensor: The result of multiplication.
        """
        # fast routine: left multiply to a numpy matrix
        if isinstance(input, np.ndarray):
            if len(input.shape) != 2 or input.shape[0] != self.shape[1]:
                raise ValueError(
                    'Cannot compute matmul(self, input): shape mismatch; '
                    'self {!r} vs input {!r}'.format(self, input)
                )
            return input[self._row_perm, :]

        # slow routine: left multiply to a TensorFlow matrix
        input = InputSpec(shape=(self.shape[1], '?')).validate('input', input)
        return tf.gather(input, indices=self._row_perm, axis=0,
                         name=name or 'left_mult')

    @add_name_arg_doc
    def right_mult(self, input, name=None):
        """
        Right multiply to `input` matrix.

        `output = matmul(input, self)`

        Args:
            input (np.ndarray or tf.Tensor): The input matrix, whose
                shape must be ``(?, self.shape[0])``.

        Returns:
            np.ndarray or tf.Tensor: The result of multiplication.
        """
        # fast routine: right multiply to a numpy matrix
        if isinstance(input, np.ndarray):
            if len(input.shape) != 2 or input.shape[1] != self.shape[0]:
                raise ValueError(
                    'Cannot compute matmul(input, self): shape mismatch; '
                    'input {!r} vs self {!r}'.format(input, self)
                )
            return input[:, self._col_perm]

        # slow routine: right multiply to a TensorFlow matrix
        input = InputSpec(shape=('?', self.shape[0])).validate('input', input)
        return tf.gather(input, indices=self._col_perm, axis=1,
                         name=name or 'right_mult')

    def inv(self):
        """
        Get the inverse permutation matrix of this matrix.

        Returns:
            PermutationMatrix: The inverse permutation matrix.
        """
        return PermutationMatrix(self._col_perm)


class InvertibleMatrix(VarScopeObject):
    """
    A matrix initialized to be an invertible, orthogonal matrix.

    If `strict` is :obj:`False`, then there is no guarantee that the matrix
    will keep invertible during training.  Otherwise, the matrix will be
    derived through a variant of PLU decomposition as proposed in
    (Kingma & Dhariwal, 2018):

    .. math::

        \\mathbf{M} = \\mathbf{P} \\mathbf{L} (\\mathbf{U} +
            \\mathrm{diag}(\\mathbf{sign} \\odot \\exp(\\mathbf{s})))

    where `P` is a permutation matrix, `L` is a lower triangular matrix
    with all its diagonal elements equal to one, `U` is an upper triangular
    matrix with all its diagonal elements equal to zero, `sign` is a vector
    of `{-1, 1}`, and `s` is a vector.  `P` and `sign` are fixed variables,
    while `L`, `U`, `s` are trainable variables.

    A `random_state` can be specified via the constructor.  If it is not
    specified, an instance of :class:`VarScopeRandomState` will be created
    according to the variable scope name of the object.
    """

    @add_name_and_scope_arg_doc
    def __init__(self, size, strict=False, dtype=tf.float32, epsilon=1e-6,
                 trainable=True, random_state=None, name=None, scope=None):
        """
        Construct a new :class:`InvertibleMatrix`.

        Args:
            size (int or (int, int)): Size of the matrix.
            strict (bool): If :obj:`True`, will derive the matrix using a
                variant of PLU decomposition, to enforce invertibility
                (see above).  If :obj:`False`, the matrix will only be
                initialized to be an orthogonal invertible matrix, without
                further constraint.  (default :obj:`False`)
            dtype (tf.DType): The data type of the variables.
            epsilon: Small float to avoid dividing by zero or taking
                logarithm of zero.
            trainable (bool): Whether or not the parameters are trainable?
            random_state (np.random.RandomState): Use this random state,
                instead of constructing a :class:`VarScopeRandomState`.
        """
        # validate the arguments
        def validate_shape():
            if is_integer(size):
                shape = (int(size),) * 2
            else:
                h, w = size
                shape = (int(h), int(w))
            if shape[0] != shape[1] or shape[0] < 1:
                raise ValueError()
            return shape

        try:
            shape = validate_shape()
        except Exception:
            raise ValueError('`size` is not valid for a square matrix: {!r}.'.
                             format(size))

        strict = bool(strict)
        dtype = tf.as_dtype(dtype)

        self._shape = shape
        self._strict = strict
        self._dtype = dtype
        self._epsilon = epsilon

        # initialize the variable scope and the random state
        super(InvertibleMatrix, self).__init__(name=name, scope=scope)
        if random_state is None:
            random_state = VarScopeRandomState(self.variable_scope)
        self._random_state = random_state

        # generate the initial orthogonal matrix
        initial_matrix = la.qr(random_state.normal(size=shape))[0]

        # create the variables
        with reopen_variable_scope(self.variable_scope):
            if not strict:
                self._matrix = model_variable(
                    'matrix',
                    initializer=tf.constant(initial_matrix, dtype=dtype),
                    dtype=dtype,
                    trainable=trainable
                )
                self._inv_matrix = tf.matrix_inverse(
                    self._matrix, name='inv_matrix')

                if is_tensorflow_version_higher_or_equal('1.10.0'):
                    self._log_det = tf.linalg.slogdet(
                        self._matrix, name='log_det')[1]
                else:
                    # low versions of TensorFlow does not have a gradient op
                    # for `slogdet`, thus we have to derive it as follows:
                    with tf.name_scope('log_det', values=[self._matrix]):
                        m = self._matrix
                        if dtype != tf.float64:
                            m = tf.cast(m, dtype=tf.float64)
                        self._log_det = tf.log(
                            tf.maximum(tf.abs(tf.matrix_determinant(m)),
                                       epsilon)
                        )
                        if self._log_det.dtype != dtype:
                            self._log_det = tf.cast(self._log_det, dtype=dtype)

                self._log_det = maybe_check_numerics(
                    self._log_det, message='numeric issues in log_det')

            else:
                initial_P, initial_L, initial_U = la.lu(initial_matrix)
                initial_s = np.diag(initial_U)
                initial_sign = np.sign(initial_s)
                initial_log_s = np.log(
                    np.maximum(np.abs(initial_s), self._epsilon))
                initial_U = np.triu(initial_U, k=1)

                # TODO: use PermutationMatrix to derive P once we can export it
                #
                # PermutationMatrix is faster, however, it cannot be exported
                # by just saving the TensorFlow variables.  Thus for the time
                # being, we have to use a true TensorFlow variable to derive P.
                #
                # P = self._P = PermutationMatrix(initial_P)

                P = self._P = model_variable(
                    'P',
                    initializer=tf.constant(initial_P, dtype=dtype),
                    dtype=dtype,
                    trainable=False
                )
                pre_L = self._pre_L = model_variable(
                    'pre_L',
                    initializer=tf.constant(initial_L, dtype=dtype),
                    dtype=dtype,
                    trainable=trainable
                )
                pre_U = self._pre_U = model_variable(
                    'pre_U',
                    initializer=tf.constant(initial_U, dtype=dtype),
                    dtype=dtype,
                    trainable=trainable
                )
                sign = self._sign = model_variable(
                    'sign',
                    initializer=tf.constant(initial_sign, dtype=dtype),
                    dtype=dtype,
                    trainable=False
                )
                log_s = self._log_s = model_variable(
                    'log_s',
                    initializer=tf.constant(initial_log_s, dtype=dtype),
                    dtype=dtype,
                    trainable=trainable
                )

                with tf.name_scope('L', values=[pre_L]):
                    L_mask = tf.constant(np.tril(np.ones(shape), k=-1),
                                         dtype=dtype)
                    L = self._L = L_mask * pre_L + tf.eye(*shape, dtype=dtype)

                with tf.name_scope('U', values=[pre_U, sign, log_s]):
                    U_mask = tf.constant(np.triu(np.ones(shape), k=1),
                                         dtype=dtype)
                    U = self._U = U_mask * pre_U + tf.diag(sign * tf.exp(log_s))

                with tf.name_scope('matrix', values=[P, L, U]):
                    self._matrix = tf.matmul(P, tf.matmul(L, U))

                with tf.name_scope('inv_matrix', values=[P, L, U]):
                    self._inv_matrix = tf.matmul(
                        tf.matrix_inverse(U, name='inv_U'),
                        tf.matmul(
                            tf.matrix_inverse(L, name='inv_L'),
                            tf.matrix_inverse(P, name='inv_P'),
                        )
                    )

                with tf.name_scope('log_det', values=[log_s]):
                    self._log_det = tf.reduce_sum(log_s)

    @property
    def shape(self):
        """
        Get the shape of the matrix.

        Returns:
            (int, int): The shape of the matrix.
        """
        return self._shape

    @property
    def matrix(self):
        """
        Get the matrix tensor.

        Returns:
            tf.Tensor or tf.Variable: The matrix tensor.
        """
        return self._matrix

    @property
    def inv_matrix(self):
        """
        Get the inverted matrix.

        Returns:
            tf.Tensor: The inverted matrix tensor.
        """
        return self._inv_matrix

    @property
    def log_det(self):
        """
        Get the log-determinant of the matrix.

        Returns:
            tf.Tensor: The log-determinant tensor.
        """
        return self._log_det
