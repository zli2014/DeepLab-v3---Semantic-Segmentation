
from Utils_101 import*
import sys

def deeplab_rn101(X, y, is_train=tf.constant(True), pkeep = 1.0):

    if pkeep != 1:
        print('\nBuilding train model')
    else:
        print('\nBuilding validation model')

    # filters = [128, 128, 256, 512, 1024]
    filters = [64, 64, 128, 256, 512]
    kernels = [7, 3, 3, 3, 3]
    strides = [2, 1, 2, 2, 4]

    size = tf.shape(X)
    height = size[1]
    width = size[2]

    # conv1
    print('\tBuilding unit: conv1')
    with tf.variable_scope('conv1'):
        x = myconv(X, kernels[0], filters[0], strides[0], name="conv")
        x = mygn(x, name="gn")
        x = myrelu(x)
        x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

    # conv2_x
    print('\tBuilding unit: conv2')
    x = first_residual_block(x, kernels[1], filters[1], strides[1], is_train, name='conv2_1')
    for i in range(2):
        name = 'conv2_' + str( i +2)
        x = residual_block(x, kernels[1], is_train, name=name)

    # conv3_x
    print('\tBuilding unit: conv3')
    x = first_residual_block(x, kernels[2], filters[2], strides[2], is_train, name='conv3_1')
    for i in range(3):
        name = 'conv3_' + str( i +2)
        x = residual_block(x, kernels[2], is_train, name=name)

    # conv4_x
    print('\tBuilding unit: conv4')
    x = first_residual_atrous_block(x, kernels[3], filters[3], strides[3], is_train, name='conv4_1')
    for i in range(22):
        name = 'conv4_' + str( i +2)
        x = residual_atrous_block(x, kernels[3], strides[3], is_train, name=name)

    multi_grid = [1, 2, 4]

    # conv5_x
    print('\tBuilding unit: conv5')
    x = first_residual_atrous_block(x, kernels[4], filters[4], strides[4]*multi_grid[0], is_train, name='conv5_1')
    for i in range(2):
        name = 'conv5_' + str( i +2)
        x = residual_atrous_block(x, kernels[4], strides[4]*multi_grid[i+1], is_train, name=name)

    # aspp
    x = atrous_spatial_pyramid_pooling_block(x, is_train, depth=256, name = 'aspp_1')

    print('\tBuilding unit: class scores')
    x = myconv(x, 1, CLASSES, 1, name="class_scores")

    # upsample logits
    print('\tBuilding unit: upsample')
    logits = tf.image.resize_images(
        x,
        tf.stack([height, width]),
        method=tf.image.ResizeMethod.BILINEAR,
        align_corners=False)

    logits_by_num_classes = tf.reshape(logits, [-1, CLASSES])

    # Probs and Predictions
    probs = tf.nn.softmax(logits, name='softmax_tensor')
    probs_by_num_classes = tf.reshape(probs, [-1, CLASSES])
    preds = tf.argmax(logits, axis=3, output_type=tf.int32)
    preds_flat = tf.reshape(preds, [-1, ])
    labels_flat = tf.reshape(y, [-1, ])

    # Remove non valid indices
    valid_indices = tf.multiply(tf.to_int32(labels_flat <= CLASSES - 1), tf.to_int32(labels_flat > -1))
    valid_probs = tf.dynamic_partition(probs_by_num_classes, valid_indices, num_partitions=2)[1]
    valid_logits = tf.dynamic_partition(logits_by_num_classes, valid_indices, num_partitions=2)[1]
    valid_labels = tf.dynamic_partition(labels_flat, valid_indices, num_partitions=2)[1]
    valid_preds = tf.dynamic_partition(preds_flat, valid_indices, num_partitions=2)[1]
    summary_histogram(valid_preds, name="valid_preds")

    # ACCURACY *****************************************************************************************************

    pixel_acc, mean_iou, per_class_acc = compute_accuracy(valid_preds, valid_labels)
    summary_scalar(pixel_acc, name="pixel_accuracy")
    summary_scalar(mean_iou, name="mean_iou")

    # COST *********************************************************************************************************

    cross_entropy = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=valid_logits,
                                                                                        labels=valid_labels,
                                                                                        name="entropy")))
    sys.stdout.flush()

    return valid_logits, valid_preds, cross_entropy, pixel_acc, mean_iou, per_class_acc
