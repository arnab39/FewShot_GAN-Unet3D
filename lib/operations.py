import tensorflow as tf

F = tf.flags.FLAGS


def gaussian_nll(mu, log_sigma, noise):
    NLL = tf.reduce_sum(log_sigma, 1) + \
              tf.reduce_sum(((noise - mu)/(1e-8 + tf.exp(log_sigma)))**2,1)/2.
    return tf.reduce_mean(NLL)


def conv3d(input_, output_dim,k_d=3, k_h=3, k_w=3, 
                  s_d=1, s_h=1, s_w=1, stddev=0.05, name="conv3d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_d, k_h, k_w, input_.get_shape()[-1], output_dim], 
                              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv3d(input_, w, strides=[1, s_d, s_h, s_w, 1], padding='SAME')
    biases = tf.get_variable('biases', [output_dim], 
                                    initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    return conv


def deconv3d(input_, output_shape,k_d=2, k_h=2, k_w=2, 
                s_d=2, s_h=2, s_w=2, stddev=0.05, name="deconv3d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_d, k_h, k_w, output_shape[-1], input_.get_shape()[-1]], 
                                    initializer=tf.random_normal_initializer(stddev=stddev))
    deconv = tf.nn.conv3d_transpose(input_, w, output_shape=output_shape, 
                                          strides=[1, s_d, s_h, s_w, 1], padding="SAME")
    biases = tf.get_variable('biases', [output_shape[-1]], 
                                            initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
    return deconv


def relu(x, name="relu"):
  return tf.nn.relu(x)

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def max_pool3D(input_,k_d=2, k_h=2, k_w=2, s_d=2, s_h=2, s_w=2):
  return tf.nn.max_pool3d(input_,[1, k_d, k_h, k_w, 1],strides=[1, s_d, s_h, s_w, 1] , padding='SAME')

def avg_pool3D(input_,k_d=2, k_h=2, k_w=2, s_d=2, s_h=2, s_w=2):
  return tf.nn.avg_pool3d(input_,[1, k_d, k_h, k_w, 1],strides=[1, s_d, s_h, s_w, 1] , padding='SAME')


def linear(input_, output_size, scope=None, stddev=0.05, bias_start=0.0):
  shape = input_.get_shape().as_list()
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
                  initializer=tf.constant_initializer(bias_start))
    return tf.matmul(input_, matrix) + bias



def instance_norm(x,phase=False,name="instance_norm"):
  epsilon = 1e-9
  mean, var = tf.nn.moments(x, [1, 2, 3], keep_dims=True)
  return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def int_shape(x):
  return list(map(int, x.get_shape()))

def get_var_maybe_avg(var_name, ema, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    v = tf.get_variable(var_name, **kwargs)
    if ema is not None:
        v = ema.average(v)
    return v

def conv3d_WN(x, num_filters, filter_size=[3,3,3], stride=[1,1,1], pad='SAME', init_scale=1., name="conv_WN", init=False, ema=None, **kwargs):
    ''' convolutional layer '''
    with tf.variable_scope(name):
        V = get_var_maybe_avg('V', ema, shape=filter_size+[int(x.get_shape()[-1]),num_filters], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        g = get_var_maybe_avg('g', ema, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        b = get_var_maybe_avg('b', ema, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        W = tf.reshape(g, [1, 1, 1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1, 2, 3])

        # calculate convolutional layer output
        x = tf.nn.bias_add(tf.nn.conv3d(x, W, [1] + stride + [1], pad), b)

        if init:  # normalize x
            m_init, v_init = tf.nn.moments(x, [0,1,2,3])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
                x = tf.identity(x)
        return x

def deconv3d_WN(x, num_filters, filter_size=[2,2,2], stride=[2,2,2], pad='SAME', init_scale=1., name="deconv_WN", init=False, ema=None, **kwargs):
    ''' transposed convolutional layer '''
    xs = int_shape(x)
    if pad=='SAME':
        target_shape = [xs[0], xs[1]*stride[0], xs[2]*stride[1], xs[3]*stride[2], num_filters]
    else:
        target_shape = [xs[0], xs[1]*stride[0] + filter_size[0]-1, xs[2]*stride[1] + filter_size[1]-1,xs[3]*stride[2] + filter_size[2]-1, num_filters]
    with tf.variable_scope(name):
        V = get_var_maybe_avg('V', ema, shape=filter_size+[num_filters,int(x.get_shape()[-1])], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        g = get_var_maybe_avg('g', ema, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        b = get_var_maybe_avg('b', ema, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        W = tf.reshape(g, [1, 1, 1, num_filters, 1]) * tf.nn.l2_normalize(V, [0, 1, 2, 4])

        # calculate convolutional layer output
        x = tf.nn.conv3d_transpose(x, W, target_shape, [1] + stride + [1], padding=pad)
        x = tf.nn.bias_add(x, b)

        if init:  # normalize x
            m_init, v_init = tf.nn.moments(x, [0,1,2,3])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
                x = tf.identity(x)
        return x

def linear_WN(x, num_units, name="linear_WN", init_scale=1., init=False, ema=None, **kwargs):
    ''' fully connected layer '''
    with tf.variable_scope(name):
        V = get_var_maybe_avg('V', ema, shape=[int(x.get_shape()[1]),num_units], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        g = get_var_maybe_avg('g', ema, shape=[num_units], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        b = get_var_maybe_avg('b', ema, shape=[num_units], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        x = tf.matmul(x, V)
        scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
        x = tf.reshape(scaler, [1, num_units]) * x + tf.reshape(b, [1, num_units])

        if init: # normalize x
            m_init, v_init = tf.nn.moments(x, [0])
            scale_init = init_scale/tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([g.assign(g*scale_init), b.assign_add(-m_init*scale_init)]):
                x = tf.identity(x)
        return x