import tensorflow as tf
import scipy.misc
import numpy as np
from sklearn.neighbors import NearestNeighbors
import PIL
from PIL import Image

BATCH_SIZE =64

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


def weight_variable(shape, name, stddev=0.02, trainable=True):
    try:
        dtype = tf.float32
        var = tf.get_variable(name, shape, tf.float32, trainable=trainable,
                              initializer=tf.random_normal_initializer(
                                  stddev=stddev, dtype=dtype))
        return var
    except ValueError as err:
        msg = "NOTE: Usually, this is due to an issue with the image dimensions.  Did you correctly set '--crop' or '--input_height' or '--output_height'?"
        err.args = err.args + (msg,)
        raise

def bias_variable(shape, name, bias_start=0.0, trainable = True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable=trainable,
                          initializer=tf.constant_initializer(
                              bias_start, dtype=dtype))
    return var


def conv2d(x, output_channels, name, k_h=5, k_w=5):
    with tf.variable_scope(name):
        x_shape = x.get_shape().as_list()
        w = weight_variable(shape=[k_h, k_w, x_shape[-1], output_channels], name='weights')
        b = bias_variable([output_channels], name='biases')
        conv = tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding='SAME') + b
        return conv


def deconv2d(x, output_shape, name, k_h=5, k_w=5):
    x_shape = x.get_shape().as_list()
    with tf.variable_scope(name):

        w = weight_variable([k_h, k_w, output_shape[-1], x_shape[-1]], name='weights')
        bias = bias_variable([output_shape[-1]], name='biases')
        deconv = tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, 2, 2, 1], padding='SAME') + bias
        return deconv

def fully_connect(x, channels_out, name):
    shape = x.get_shape().as_list()
    channels_in = shape[1]
    with tf.variable_scope(name):
        weights = weight_variable([channels_in, channels_out], name='weights')
        biases = bias_variable([channels_out], name='biases')
        return tf.matmul(x, weights) + biases

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def conv_cond_concat(value, cond):
    value_shapes = value.get_shape().as_list()
    cond_shapes = cond.get_shape().as_list()
    return tf.concat([value, cond * tf.ones(value_shapes[0:3] + cond_shapes[3:])], 3)


def relu(value):
    return tf.nn.relu(value)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
	shape = input_.get_shape().as_list()

	with tf.variable_scope(scope or "Linear"):
		matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
								 tf.random_normal_initializer(stddev=stddev))
		bias = tf.get_variable("bias", [output_size],
			initializer=tf.constant_initializer(bias_start))
		if with_w:
			return tf.matmul(input_, matrix) + bias, matrix, bias
		else:
			return tf.matmul(input_, matrix) + bias


def generator(z, y, training=True):
    s = 64
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

    g_bn0 = batch_norm(name='g_bn0')
    g_bn1 = batch_norm(name='g_bn1')
    g_bn2 = batch_norm(name='g_bn2')
    g_bn3 = batch_norm(name='g_bn3')
    reduced_text_embedding =lrelu( linear(y, 100, 'g_embedding') )
    z_concat = tf.concat([z, reduced_text_embedding],1)
    z_ = linear(z_concat, 128*8*s16*s16, 'g_h0_lin')
    h0 = tf.reshape(z_, [-1, s16, s16,128 * 8])
    h0 = tf.nn.relu(g_bn0(h0))

    h1 = deconv2d(h0, [BATCH_SIZE, s8, s8, 128*4], name='g_h1')
    h1 = tf.nn.relu(g_bn1(h1))

    h2 = deconv2d(h1, [BATCH_SIZE, s4, s4, 128*2], name='g_h2')
    h2 = tf.nn.relu(g_bn2(h2))

    h3 = deconv2d(h2, [BATCH_SIZE, s2, s2, 128*1], name='g_h3')
    h3 = tf.nn.relu(g_bn3(h3))

    h4 = deconv2d(h3, [BATCH_SIZE, s, s, 3], name='g_h4')

    return tf.nn.tanh(h4)

def discriminator(image, t_text_embedding, reuse=False, training=True):
    # with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
    if reuse:
            tf.get_variable_scope().reuse_variables()
    d_bn1 = batch_norm(name='d_bn1')
    d_bn2 = batch_norm(name='d_bn2')
    d_bn3 = batch_norm(name='d_bn3')
    d_bn4 = batch_norm(name='d_bn4')
    h0 = lrelu(conv2d(image, 128, name = 'd_h0_conv')) #32
    h1 = lrelu( d_bn1(conv2d(h0, 128*2, name = 'd_h1_conv'))) #16
    h2 = lrelu( d_bn2(conv2d(h1, 128*4, name = 'd_h2_conv'))) #8
    h3 = lrelu( d_bn3(conv2d(h2, 128*8, name = 'd_h3_conv'))) #4

	# ADD TEXT EMBEDDING TO THE NETWORK
    reduced_text_embeddings = lrelu(linear(t_text_embedding, 100, 'd_embedding'))
    reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,1)
    reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,2)
    tiled_embeddings = tf.tile(reduced_text_embeddings, [1,4,4,1], name='tiled_embeddings')

    h3_concat = tf.concat([h3, tiled_embeddings],3, name='h3_concat')
    h3_new = lrelu( d_bn4(conv2d(h3_concat, 128*8,name = 'd_h3_conv_new'))) #4

    h4 = linear(tf.reshape(h3_new, [BATCH_SIZE, -1]), 1, 'd_h3_lin')

    return tf.nn.sigmoid(h4), h4

def sampler(z, y, training=False):
    tf.get_variable_scope().reuse_variables()
    return generator(z, y, training=training)

def save_images(images, size, path):

    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]

    merge_img = np.zeros((h * size[0], w * size[1], 3))


    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        if j >= size[0]:
            break
        merge_img[j * h:j * h + h, i * w:i * w + w, :] = image


    return scipy.misc.imsave(path, merge_img)


def read_data(data):
    values = list(data.values())
    keys = list(data.keys())  # BATCH_SIZE is the windows size

    pairs = list(zip(keys,values))
    np.random.shuffle(pairs)
    keys,values = zip(*pairs)

    img=scipy.misc.imread("./selected/"+'/'+keys[0].split('*')[0]+".jpg").astype(np.float)
    img=transform(img,128,128,64,64,True)
    img=np.asarray(img).reshape(1,-1)
    labels=values[0].reshape(1,-1)

    for i,j in zip(keys[1:],values[1:]):
        try:

            img_o=scipy.misc.imread("./selected/"+'/'+i.split('*')[0]+".jpg").astype(np.float)
            img_o=transform(img_o,128,128,64,64,True)
            img_o=np.asarray(img_o).reshape(1,-1)
            img=np.concatenate([img,img_o],axis=0)
            labels=np.concatenate([labels,j.reshape(1,-1)],axis=0)
        except (FileNotFoundError,IOError):
            print("Error:"+i)
            pass

    return [img,labels]

def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(
      image, input_height, input_width,
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])
