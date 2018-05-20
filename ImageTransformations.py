
import pdb
import pkg_resources
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from UsefulFunctions import*
from GlobalVariables import*
#import matplotlib.pyplot as plt
import numpy as np
import pdb

def random_rotation_image_with_annotation(image_tensor, annotation_tensor, max_angle):

    # Random variable: two possible outcomes (0 or 1)
    # with 0.5 chance
    random_var = tf.cast(tf.random_uniform(maxval=2, dtype=tf.int32, shape=[]),dtype=tf.float32)

    # Random selection of angle and direction of rotation
    random_angle = tf.cast(tf.random_uniform(maxval=max_angle, dtype=tf.int32, shape=[]),dtype=tf.float32)
    random_direction = tf.cast(tf.random_uniform(minval=-1, maxval=1, dtype=tf.int32, shape=[]),dtype=tf.float32)
    randomly_rotated_img = control_flow_ops.cond(pred=tf.equal(tf.multiply(tf.abs(random_direction), random_var), 0),
                                                 true_fn=lambda: tf.contrib.image.rotate(image_tensor,
                                                                                         random_direction * random_angle,
                                                                                         interpolation='NEAREST'),
                                                 false_fn=lambda: image_tensor)
    randomly_rotated_annotation = control_flow_ops.cond(pred=tf.equal(tf.multiply(tf.abs(random_direction), random_var), 0),
                                                 true_fn=lambda: tf.contrib.image.rotate(annotation_tensor,
                                                                                         random_direction * random_angle,
                                                                                         interpolation='NEAREST'),
                                                 false_fn=lambda: annotation_tensor)

    return randomly_rotated_img, randomly_rotated_annotation

def flip_randomly_left_right_image_with_annotation(image_tensor, annotation_tensor):
    """Accepts image tensor and annotation tensor and returns randomly flipped tensors of both.
    The function performs random flip of image and annotation tensors with probability of 1/2
    The flip is performed or not performed for image and annotation consistently, so that
    annotation matches the image.

    Parameters
    ----------
    image_tensor : Tensor of size (width, height, 3)
        Tensor with image
    annotation_tensor : Tensor of size (width, height, 1)
        Tensor with annotation

    Returns
    -------
    randomly_flipped_img : Tensor of size (width, height, 3) of type tf.float.
        Randomly flipped image tensor
    randomly_flipped_annotation : Tensor of size (width, height, 1)
        Randomly flipped annotation tensor

    """

    # Random variable: two possible outcomes (0 or 1)
    # with 0.5 chance
    random_var = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])


    randomly_flipped_img = control_flow_ops.cond(pred=tf.equal(random_var, 0),
                                                 true_fn=lambda: tf.image.flip_left_right(image_tensor),
                                                 false_fn=lambda: image_tensor)

    randomly_flipped_annotation = control_flow_ops.cond(pred=tf.equal(random_var, 0),
                                                        true_fn=lambda: tf.image.flip_left_right(annotation_tensor),
                                                        false_fn=lambda: annotation_tensor)

    return randomly_flipped_img, randomly_flipped_annotation

def flip_randomly_up_down_image_with_annotation(image_tensor, annotation_tensor):
    """Accepts image tensor and annotation tensor and returns randomly flipped tensors of both.
       The function performs random flip of image and annotation tensors with probability of 1/2
       The flip is performed or not performed for image and annotation consistently, so that
       annotation matches the image.

       Parameters
       ----------
       image_tensor : Tensor of size (width, height, 3)
           Tensor with image
       annotation_tensor : Tensor of size (width, height, 1)
           Tensor with annotation

       Returns
       -------
       randomly_flipped_img : Tensor of size (width, height, 3) of type tf.float.
           Randomly flipped image tensor
       randomly_flipped_annotation : Tensor of size (width, height, 1)
           Randomly flipped annotation tensor

       """

    # Random variable: two possible outcomes (0 or 1)
    # with 0.5 chance
    random_var = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])

    randomly_flipped_img = control_flow_ops.cond(pred=tf.equal(random_var, 0),
                                                 true_fn=lambda: tf.image.flip_up_down(image_tensor),
                                                 false_fn=lambda: image_tensor)

    randomly_flipped_annotation = control_flow_ops.cond(pred=tf.equal(random_var, 0),
                                                        true_fn=lambda: tf.image.flip_up_down(annotation_tensor),
                                                        false_fn=lambda: annotation_tensor)

    return randomly_flipped_img, randomly_flipped_annotation

def random_color_distortion(image_tensor, annotation_tensor):

    random_var_brightness = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])
    distorted_image = control_flow_ops.cond(pred=tf.equal(random_var_brightness, 0),
                                                true_fn=lambda: tf.image.random_brightness(image_tensor, max_delta=32. / 255.),
                                                false_fn=lambda: image_tensor)
    random_var_saturation = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])
    distorted_image = control_flow_ops.cond(pred=tf.equal(random_var_saturation, 0),
                                            true_fn=lambda: tf.image.random_saturation(distorted_image, lower=0.5, upper=1.5),
                                            false_fn=lambda: distorted_image)
    random_var_hue = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])
    distorted_image = control_flow_ops.cond(pred=tf.equal(random_var_hue, 0),
                                            true_fn=lambda: tf.image.random_hue(distorted_image, max_delta=0.2),


                                            false_fn=lambda: distorted_image)
    random_var_contrast = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])
    distorted_image = control_flow_ops.cond(pred=tf.equal(random_var_contrast, 0),
                                            true_fn=lambda: tf.image.random_contrast(distorted_image, lower=0.5, upper=1.5),
                                            false_fn=lambda: distorted_image)

    return tf.clip_by_value(distorted_image, 0.0, 1.0), annotation_tensor