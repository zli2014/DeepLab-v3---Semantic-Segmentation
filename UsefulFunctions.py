
# IMPORTS

import tensorflow as tf
from GlobalVariables import*
import numpy as np
import pdb

# COLLECTION KEYS ******************************************************************************************************

WEIGHT_DECAY_KEY = 'WEIGHT_DECAY' # l2 norm collection

# LAYER FUNCTIONS ******************************************************************************************************

def myconv(x, filter_size, out_channel, strides, pad='SAME', name='conv'):
    # Step 1: create filter variable
    in_shape = x.get_shape()
    with tf.variable_scope(name):
        # Main operation: conv2d
        with tf.device('/CPU:0'):
            kernel = tf.get_variable('kernel', [filter_size, filter_size, in_shape[3], out_channel],
                                     tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / filter_size / filter_size / out_channel)))
            bias = tf.get_variable('bias', [out_channel], tf.float32, initializer=tf.zeros_initializer())
    # Step 2: add variables to the list of l2 normalized variables
        if kernel not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, kernel)
        if bias not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, bias)
    # Step 3: convolution
    conv = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], pad) + bias
    return conv

def myatrouconv(x, filter_size, out_channel, output_stride, pad='SAME', name='atrous_conv'):
    # Step 1: create filter variable
    in_shape = x.get_shape()
    with tf.variable_scope(name):
        with tf.device('/CPU:0'):
            #kernel = tf.Variable(identity_initializer([filter_size, filter_size, in_channels, out_channel]),
            #                         tf.float32, name='kernel')
            #bias = tf.get_variable('bias', [out_channel], tf.float32, initializer=tf.zeros_initializer())
            kernel = tf.get_variable('kernel', [filter_size, filter_size, in_shape[3], out_channel],
                                     tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / filter_size / filter_size / out_channel)))
        bias = tf.get_variable('bias', [out_channel], tf.float32, initializer=tf.zeros_initializer())
    # Step 2: add variables to the list of l2 normalized variables
        if kernel not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, kernel)
        if bias not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, bias)
    # Step 3: convolution
    atrous = tf.nn.atrous_conv2d(x, kernel, rate=output_stride, padding=pad) + bias
    return  atrous

def mybn(x, is_train, name='bn'):
    moving_average_decay = 0.9
    with tf.variable_scope(name):
        decay = moving_average_decay
        # Get batch mean and var, which will be used during training
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
        # Define variables, mu and sigma are not trainable since depend on the batch (train) or the population (test)
        with tf.device('/CPU:0'):
            mu = tf.get_variable('mu', batch_mean.get_shape(), tf.float32,
                                 initializer=tf.zeros_initializer(), trainable=False)
            sigma = tf.get_variable('sigma', batch_var.get_shape(), tf.float32,
                                    initializer=tf.ones_initializer(), trainable=False)
            beta = tf.get_variable('beta', batch_mean.get_shape(), tf.float32,
                                   initializer=tf.zeros_initializer())
            gamma = tf.get_variable('gamma', batch_var.get_shape(), tf.float32,
                                    initializer=tf.ones_initializer())
        update = 1.0 - decay
        update_mu = mu.assign_sub(update * (mu - batch_mean))
        update_sigma = sigma.assign_sub(update * (sigma - batch_var))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)
        mean, var = tf.cond(is_train, lambda: (batch_mean, batch_var),
                            lambda: (mu, sigma))
        bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
    return bn

def myrelu(x, leakness=0.0, name=None):
    if leakness > 0.0:
        name = 'lrelu' if name is None else name
        return tf.maximum(x, x*leakness, name='lrelu')
    else:
        name = 'relu' if name is None else name
        return tf.nn.relu(x, name='relu')

# RESNET MODULES *******************************************************************************************************

def first_residual_block(x, kernel, out_channel, strides, is_train, name="unit"):
    input_channels = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        print('\tBuilding residual unit: %s' % scope.name)
        # Shortcut connection
        if input_channels == out_channel:
            if strides == 1:
                shortcut = tf.identity(x) # returns a tensor with the same shape and contents as x
            else:
                shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
        else:
            shortcut = myconv(x, 1, out_channel, strides, name='shortcut') # 1x1 conv to obtain out_channel maps
        # Residual
        x = myconv(x, kernel, out_channel, strides, name='conv_1')
        x = mybn(x, is_train, name='bn_1')
        x = myrelu(x, name='relu_1')
        x = myconv(x, kernel, out_channel, 1, name='conv_2')
        x = mybn(x, is_train, name='bn_2')
        # Merge
        x = x + shortcut
        x = myrelu(x, name='relu_2')
    return x

def residual_block(x, kernel, is_train, name="unit"):
    num_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        print('\tBuilding residual unit: %s' % scope.name)
        # Shortcut connection
        shortcut = x
        # Residual
        x = myconv(x, kernel, num_channel, 1, name='conv_1')
        x = mybn(x, is_train, name='bn_1')
        x = myrelu(x, name='relu_1')
        x = myconv(x, kernel, num_channel, 1, name='conv_2')
        x = mybn(x, is_train, name='bn_2')
        # Merge
        x = x + shortcut
        x = myrelu(x, name='relu_2')
    return x

# DEEPLAB MODULES ******************************************************************************************************

def first_residual_atrous_block(x, kernel, out_channel, strides, is_train, name="unit"):
    input_channels = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        print('\tBuilding residual unit: %s' % scope.name)
        # Shortcut connection
        if input_channels == out_channel:
            if strides == 1:
                shortcut = tf.identity(x) # returns a tensor with the same shape and contents as x
            else:
                shortcut = tf.nn.max_pool(x, [1, 1, 1, 1], [1, 1, 1, 1], 'VALID')
        else:
            shortcut = myatrouconv(x, 1, out_channel, strides, name='shortcut') # 1x1 conv to obtain out_channel maps
        # Residual
        x = myatrouconv(x, kernel, out_channel, strides, name='conv_1')
        x = mybn(x, is_train, name='bn_1')
        x = myrelu(x, name='relu_1')
        x = myatrouconv(x, kernel, out_channel, strides, name='conv_2')
        x = mybn(x, is_train, name='bn_2')
        # Merge
        x = x + shortcut
        x = myrelu(x, name='relu_2')
    return x

def residual_atrous_block(x, kernel, is_train, name="unit"):
    num_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        print('\tBuilding residual unit: %s' % scope.name)
        # Shortcut connection
        shortcut = x
        # Residual
        x = myatrouconv(x, kernel, num_channel, 2, name='conv_1')
        x = mybn(x, is_train, name='bn_1')
        x = myrelu(x, name='relu_1')
        x = myatrouconv(x, kernel, num_channel, 2, name='conv_2')
        x = mybn(x, is_train, name='bn_2')
        # Merge
        x = x + shortcut
        x = myrelu(x, name='relu_2')
    return x

def atrous_spatial_pyramid_pooling_block(x, is_train, depth=256, name = 'aspp'):
    input_size = tf.shape(x)[1:3]
    filters = [1, 4, 3, 3, 1, 1]
    atrous_rates = [1, 6, 12, 18, 1, 1]
    with tf.variable_scope(name) as scope:
        print('\tBuilding aspp unit: %s' % scope.name)
        # Branch 0: 1x1 conv
        branch0 = myconv(x, filters[0], depth, atrous_rates[0], name='branch0')
        branch0 = mybn(branch0, is_train, name='bn_0')
        # Branch 1: 3x3 atrous_conv (rate = 6)
        branch1 = myatrouconv(x, filters[1], depth, atrous_rates[1], name='branch1')
        branch1 = mybn(branch1, is_train, name='bn_1')
        # Branch 2: 3x3 atrous_conv (rate = 12)
        branch2 = myatrouconv(x, filters[2], depth, atrous_rates[2], name='branch2')
        branch2 = mybn(branch2, is_train, name='bn_2')
        # Branch 3: 3x3 atrous_conv (rate = 18)
        branch3 = myatrouconv(x, filters[3], depth, atrous_rates[3], name='branch3')
        branch3 = mybn(branch3, is_train, name='bn_3')
        # Branch 4: image pooling
        # 4.1 global average pooling
        branch4 = tf.reduce_mean(x, [1, 2], name='global_average_pooling', keepdims=True)
        # 4.2 1x1 convolution with 256 filters and batch normalization
        branch4 = myconv(x, filters[4], depth, atrous_rates[4], name='brach4')
        branch4 = mybn(branch4, is_train, name='bn_4')
        # 4.3 bilinearly upsample features
        branch4 = tf.image.resize_bilinear(branch4, input_size, name='branch4_upsample')
        # Output
        out = tf.concat([branch0, branch1, branch2, branch3, branch4], axis=3, name='aspp_concat')
        out = myconv(out, filters[5], depth, atrous_rates[5], name='aspp_out')
        return out

# INITIALIZER FUNCTIONS ************************************************************************************************

def identity_initializer(filter_shape):

    """returns the values of a filter that simply passes forward the input feature map"""

    filter = np.zeros((filter_shape), dtype='float32')
    center = filter_shape[1]/2
    filter[ center, center, center, :] = np.ones((filter_shape[3]),dtype='float32')
    return filter

    #filter = np.zeros((filter_shape), dtype='float32')
    #center = filter_shape[1]/2
    #for i in range(filter_shape[2]):
    #        filter[center, center, i, i] = np.float(1)
    #return filter

# ACCURACY FUNCTIONS ***************************************************************************************************

def compute_accuracy(valid_preds, valid_labels):

    """Computes both pixel accuracy and mean intersection-over-union accuracy"""

    train_pixel_acc = tf.metrics.accuracy(valid_labels, valid_preds)
    mean_iou = tf.metrics.mean_iou(valid_labels, valid_preds, CLASSES)
    train_mean_iou = compute_mean_iou(mean_iou[1])
    return train_pixel_acc[1], train_mean_iou

def compute_mean_iou(total_cm, name='mean_iou'):

    """Compute the mean intersection-over-union via the confusion matrix."""

    sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
    sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
    cm_diag = tf.to_float(tf.diag_part(total_cm))
    denominator = sum_over_row + sum_over_col - cm_diag

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = tf.reduce_sum(tf.cast(
        tf.not_equal(denominator, 0), dtype=tf.float32))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = tf.where(
        tf.greater(denominator, 0),
        denominator,
        tf.ones_like(denominator))
    iou = tf.div(cm_diag, denominator)

    for i in range(CLASSES):
        tf.identity(iou[i], name='train_iou_class{}'.format(i))
        tf.summary.scalar('train_iou_class{}'.format(i), iou[i])

    # If the number of valid entries is 0 (no classes) we return 0.
    result = tf.where(
        tf.greater(num_valid_entries, 0),
        tf.reduce_sum(iou, name=name) / num_valid_entries,
        0)
    return result

# LOAD IMAGES **********************************************************************************************************

def read_and_decode(data_path, epochs, batch_size):

    with tf.Session() as sess:
        feature = {'image': tf.FixedLenFeature([], tf.string),
                   'label': tf.FixedLenFeature([], tf.string)}

        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer([data_path], num_epochs=epochs)

        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)

        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features['image'], tf.float32)
        label = tf.decode_raw(features['label'], tf.float32)

        # Reshape image data into the original shape
        image = tf.reshape(image, [SEGMENTATION_HEIGHT, SEGMENTATION_WIDTH, 3])
        label = tf.reshape(label, [SEGMENTATION_HEIGHT, SEGMENTATION_HEIGHT, 1])

        resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                               target_height=SEGMENTATION_HEIGHT,
                                                               target_width=SEGMENTATION_WIDTH)
        resized_label = tf.image.resize_image_with_crop_or_pad(image=label,
                                                               target_height=SEGMENTATION_HEIGHT,
                                                               target_width=SEGMENTATION_WIDTH)

        # Creates batches by randomly shuffling tensors
        images, labels = tf.train.shuffle_batch([resized_image, resized_label], batch_size=batch_size, capacity=30,
                                                num_threads=1, min_after_dequeue=10, allow_smaller_final_batch=True)

        return images, labels

# DECAY FUNCTIONS ******************************************************************************************************

def lr_decay(learning_rate):
    return (learning_rate * LR_DECAY)

# PREPROCESS DATA ******************************************************************************************************

def preprocess_data(batch_X, batch_Y):
    train_mean = np.load(DATASET_PATH + 'train_mean.npy')
    train_mean = np.reshape(train_mean,[1,SEGMENTATION_HEIGHT,SEGMENTATION_WIDTH,3])
    batch_X = np.subtract(batch_X, train_mean)
    padding_margin = (IMAGE_HEIGHT - SEGMENTATION_HEIGHT) / 2
    batch_X = np.pad(batch_X, ((0,0),(padding_margin,padding_margin),(padding_margin,padding_margin),(0,0)), 'symmetric')
    batch_X = batch_X / 255
    batch_Y = np.subtract(batch_Y, np.ones((batch_Y.shape[0], SEGMENTATION_HEIGHT, SEGMENTATION_WIDTH, 1)) * 92)
    return batch_X, batch_Y

# SUMMARY FUNCTIONS ******************************************************************************************************

def summary_scalar(tensor_scalar, name):
    if DEBUG == 1:
        tf.identity(tensor_scalar, name=name)
        tf.summary.scalar(name, tensor_scalar)

def summary_histogram(tensor_histogram, name):
    if DEBUG == 1:
        tf.identity(tensor_histogram, name=name)
        tf.summary.histogram(name, tensor_histogram)

def summary_text(tensor_text, name):
    if DEBUG == 1:
        tf.identity(tensor_text, name=name)
        tf.summary.text(name, tensor_text)
