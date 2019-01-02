from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework.ops import convert_to_tensor


class MLESNCell(rnn_cell_impl.RNNCell):

  def __init__(self, num_units, wr2_scale=0.7, connectivity=0.1, leaky=1.0, activation=math_ops.tanh,
               win_init=init_ops.random_normal_initializer(),
               wr_init=init_ops.random_normal_initializer(),
               bias_init=init_ops.random_normal_initializer()):
    self._num_units = num_units
    self._leaky = leaky
    self._activation = activation

    def _wr_initializer(shape, dtype, partition_info=None):
      wr = wr_init(shape, dtype=dtype)

      connectivity_mask = math_ops.cast(
          math_ops.less_equal(
            random_ops.random_uniform(shape),
            connectivity),
        dtype)

      wr = math_ops.multiply(wr, connectivity_mask)

      wr_norm2 = math_ops.sqrt(math_ops.reduce_sum(math_ops.square(wr)))

      is_norm_0 = math_ops.cast(math_ops.equal(wr_norm2, 0), dtype)

      wr = wr * wr2_scale / (wr_norm2 + 1 * is_norm_0)

      return wr

    self._win_initializer = win_init
    self._bias_initializer = bias_init
    self._wr_initializer = _wr_initializer

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):

    inputs = convert_to_tensor(inputs)
    input_size = inputs.get_shape().as_list()[1]
    dtype = inputs.dtype

    with vs.variable_scope(scope or type(self).__name__):  # "ESNCell"

      # Layer 0

      win = vs.get_variable("InputMatrix", [input_size, self._num_units], dtype=dtype,
                            trainable=False, initializer=self._win_initializer)
      wr = vs.get_variable("ReservoirMatrix", [self._num_units, self._num_units], dtype=dtype,
                           trainable=False, initializer=self._wr_initializer)
      # Layer 1

      win_l1 = vs.get_variable("InputMatrix_L1",[self._num_units, self._num_units], dtype=dtype,
                            trainable=False, initializer=self._win_initializer)

      wr_l1 = vs.get_variable("ReservoirMatrix_L1",[self._num_units,self._num_units],dtype=dtype,
                              trainable=False, initializer=self._wr_initializer)
      # .
      # .
      # .
      # Layer N

      b = vs.get_variable("Bias", [self._num_units], dtype=dtype, trainable=False, initializer=self._bias_initializer)

      in_mat = array_ops.concat([inputs, state], axis=1)
      weights_mat = array_ops.concat([win, wr], axis=0)

      x = (1 - self._leaky) * state + self._leaky * self._activation(math_ops.matmul(in_mat, weights_mat) + b)

      in_mat = array_ops.concat([x,state], axis=1)
      weights_mat = array_ops.concat([win_l1,wr_l1], axis=0)

      x = (1 - self._leaky) * state + self._leaky * self._activation(math_ops.matmul(in_mat, weights_mat) + b)

    return x, x
