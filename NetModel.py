#https://github.com/dalgu90/resnet-18-tensorflow

# IMPORTS

from GlobalVariables import*
from UsefulFunctions import*
from datetime import datetime
from hyperopt import hp, fmin, tpe
import sys
import os

# RUN OPTIONS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)

FLAGS = tf.app.flags.FLAGS

# NETWORK CLASS

class deepLabNet:

    # DEFINE CLASS *****************************************************************************************************

    def __init__(self):

        self.lr = tf.placeholder(tf.float32)

    # BUILD UP THE MODEL ***********************************************************************************************

    def build_model(self, X, y, is_train=tf.constant(True), pkeep = 1): # ResNet 18

        if pkeep != 1:
            print('\nBuilding train model')
        else:
            print('\nBuilding validation model')

        # filters = [128, 128, 256, 512, 1024]
        filters = [64, 64, 128, 256, 512]
        kernels = [7, 3, 3, 3, 3]
        strides = [2, 0, 2, 2, 2]

        size = tf.shape(X)
        height = size[1]
        width = size[2]

        # conv1
        print('\tBuilding unit: conv1')
        with tf.variable_scope('conv1'):
            x = myconv(X, kernels[0], filters[0], strides[0], name="conv")
            x = mybn(x, is_train, name="bn")
            x = myrelu(x)
            self.shape_conv1 = tf.shape(x)
            x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

        # conv2_x
        x = residual_block(x,kernels[1], is_train, name='conv2_1')
        x = residual_block(x, kernels[1], is_train, name='conv2_2')
        self.shape_conv2 = tf.shape(x)

        # conv3_x
        x = first_residual_block(x, kernels[2], filters[2], strides[2], is_train, name='conv3_1')
        x = residual_block(x, kernels[2], is_train, name='conv3_2')
        self.shape_conv3 = tf.shape(x)

        # conv4_x
        x = first_residual_block(x, kernels[3], filters[3], strides[3], is_train, name='conv4_1')
        x = residual_block(x, kernels[3], is_train, name='conv4_2')
        self.shape_conv4 = tf.shape(x)

        # conv5_x
        x = first_residual_atrous_block(x, kernels[4], filters[4], strides[4], is_train, name='conv5_1')
        x = residual_atrous_block(x, kernels[4], is_train, name='conv5_2')
        self.shape_conv5 = tf.shape(x)

        # aspp
        x = atrous_spatial_pyramid_pooling_block(x, is_train, depth=256, name = 'aspp_1')
        self.shape_aspp = tf.shape(x)

        print('\tBuilding unit: class scores') # Maybe another layer ???
        x = myconv(x, 1, CLASSES, 1, name="class_scores")
        self.shape_class_scores = tf.shape(x)

        # upsample logits
        print('\tBuilding unit: upsample')
        logits = tf.image.resize_images(
            x,
            tf.stack([height, width]),
            method=tf.image.ResizeMethod.BILINEAR,
            align_corners=False)

        self.shape_upsampled_logits = tf.shape(logits)

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

        self.pixel_acc, self.mean_iou, self.per_class_acc = compute_accuracy(valid_preds, valid_labels)
        summary_scalar(self.pixel_acc, name="pixel_accuracy")
        summary_scalar(self.mean_iou, name="mean_iou")

        # COST *********************************************************************************************************

        self.cross_entropy = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=valid_logits,
                                                                              labels=valid_labels,
                                                                              name="entropy")))
        sys.stdout.flush()

        return valid_logits, valid_preds, self.cross_entropy, self.pixel_acc, self.mean_iou, self.per_class_acc

    # BUILD TRAIN OPERATION ********************************************************************************************

    def build_train_op(self, X, y, is_train, pkeep):

        # Build Model
        logits, preds, loss, pixel_acc, mean_iou, per_class_acc = self.build_model(X, y, is_train, pkeep)

        # Learning rate - save in summary
        tf.summary.scalar('learing_rate', self.lr)

        # Train step
        opt = tf.train.MomentumOptimizer(self.lr, MOMENTUM)

        # Add l2 loss
        costs = [tf.nn.l2_loss(var) for var in tf.get_collection(WEIGHT_DECAY_KEY)]
        l2_loss = tf.multiply(WEIGHT_DECAY, tf.add_n(costs))
        total_loss = loss + l2_loss

        # Loss and total loss - save in summary
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('total_loss', total_loss)

        # Compute gradients of total loss
        grads_and_vars = opt.compute_gradients(total_loss, tf.trainable_variables())

        for grad, var in grads_and_vars:
            if grad is not None:
                grad_name = var.op.name + "/gradient"
                summary_histogram(grad, name=grad_name)

        # Apply gradient
        apply_grad_op = opt.apply_gradients(grads_and_vars, global_step=None)

        # Batch normalization moving average update
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(*(update_ops + [apply_grad_op])) # runs all operarions

        print("Setting up summary op")
        summary_op = tf.summary.merge_all()

        return loss, pixel_acc, mean_iou, per_class_acc, train_op, summary_op

    # TRAIN EPOCH ******************************************************************************************************

    def train_epoch(self, sess, current_learning_rate):

        sess.run(self.reset_op)
        epoch_loss = 0
        steps = int(float(TRAIN_IMAGES) * (float(CROPS_PER_IMAGE) / float(BATCH_SIZE)))

        for i in range(steps):
            feed_dict = {self.lr: current_learning_rate}
            tr_loss, tr_pixel_acc, tr_mean_iou_acc, tr_mean_per_class_acc, _ = sess.run(
                [self.train_loss, self.train_pixel_acc, self.train_mean_iou_acc, self.train_mean_per_class_acc, self.train_step],
                feed_dict=feed_dict, options=run_options)
            if np.isnan(tr_loss):
                print('got NaN as the loss value for 1 image')
            else:
                epoch_loss += tr_loss

            if DEBUG:
                summary_str = sess.run([self.summary_op], feed_dict=feed_dict, options=run_options)
                self.summary_writer.add_summary(summary_str)

        epoch_loss = epoch_loss / steps
        return epoch_loss, tr_pixel_acc, tr_mean_iou_acc, tr_mean_per_class_acc

    # GET VAL COST/ACC *************************************************************************************************

    def evaluate_model_on_val_set(self, sess):

        sess.run(self.reset_op)
        epoch_loss = 0
        steps = int(float(VALIDATION_IMAGES) * (float(CROPS_PER_IMAGE) / float(BATCH_SIZE)))

        for i in range(steps):
            v_loss, v_pixel_acc, v_mean_iou_acc, v_mean_per_class_acc = \
                sess.run([self.val_loss, self.val_pixel_acc, self.val_mean_iou_acc, self.val_mean_per_class_acc], options=run_options)
            if np.isnan(v_loss):
                print('got NaN as the loss value for 1 image')
            else:
                epoch_loss += v_loss

        epoch_loss = epoch_loss / steps
        return epoch_loss, v_pixel_acc, v_mean_iou_acc, v_mean_per_class_acc

    # TRAIN MODEL ******************************************************************************************************

    def train(self, args):

        learning_rate, dropout, epochs = args
        initial_learning_rate = learning_rate
        epochs = int(epochs)

        # Get images and labels
        with tf.device('/cpu:0'):
            train_images, train_labels = read_and_decode(DATASET_PATH + 'train_dataset.tfrecords', epochs, BATCH_SIZE)
            val_images, val_labels = read_and_decode(DATASET_PATH + 'validation_dataset.tfrecords', epochs, BATCH_SIZE)

        # Define both train and val models
        with tf.variable_scope("model"):
            self.train_loss, self.train_pixel_acc, self.train_mean_iou_acc, self.train_mean_per_class_acc, self.train_step, self.summary_op = \
                self.build_train_op(X=train_images, y=train_labels, pkeep=dropout, is_train=tf.constant(True))
        with tf.variable_scope("model", reuse=True):
            self.val_loss, self.val_pixel_acc, self.val_mean_iou_acc, self.val_mean_per_class_acc, _, _ = \
                self.build_train_op(X=val_images, y=val_labels, pkeep=1, is_train=tf.constant(False))

        # Initialization
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess = tf.Session()
        sess.run(init_op)
        stream_vars = [i for i in tf.local_variables() if i.name.split('/')[1] == 'accuracy']
        self.reset_op = [tf.initialize_variables(stream_vars)]

        # Load Imagenet Pretrained Weights
        print("\nLoading ResNet18 pretrained weights")
        resnet18_weights = np.load("resnet18_weights.npy")
        all_vars = tf.trainable_variables()
        notloaded = []
        print('Successful:')
        for v in all_vars:
            try:
                assign_op = v.assign(resnet18_weights.item().get(v.op.name[6:]))
                sess.run(assign_op)
                print('\t' + v.op.name)
            except Exception, e:
                notloaded.append(v.op.name)
        print('Unsuccessful:')
        for name in notloaded:
            print('\t'+name)
        sys.stdout.flush()

        # Create a saver
        saver = tf.train.Saver(max_to_keep=100)
        if RESTORE is not 0:
            saver.restore(sess, RESTORE_PATH)
            print('Load checkpoint %s' % RESTORE_PATH)
            sys.stdout.flush()
        else:
            print('\nNo checkpoint file of basemodel found. Start from the scratch.\n')
            sys.stdout.flush()

        # Create summary writer
        if DEBUG == 1:
            self.summary_writer = tf.summary.FileWriter(LOGS_DIR, sess.graph)

        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        train_cost = []
        val_cost = []
        train_accuracy_pix = []
        train_accuracy_iou = []
        train_accuracy_per_class = []
        val_accuracy_pix = []
        val_accuracy_iou = []
        val_accuracy_per_class = []

        print("Starting to train...")
        sys.stdout.flush()

        try:
            epoch = 1
            lowest_val_loss_epoch = 0
            best_val_loss = float('inf')
            patience = 0
            while (True):

                # Check start time
                start = datetime.now()

                # Train for 1 epoch on training set
                tc, ta_pix, ta_iou, ta_class = self.train_epoch(sess, learning_rate)
                ta_class = np.mean(ta_class[np.nonzero(ta_class)])
                train_cost.append(tc)
                train_accuracy_pix.append(ta_pix)
                train_accuracy_iou.append(ta_iou)
                train_accuracy_per_class.append(ta_class)

                # Compute cost/accuracy on val dataset
                vc, va_pix, va_iou, va_class = self.evaluate_model_on_val_set(sess)
                va_class = np.mean(va_class[np.nonzero(va_class)])
                val_cost.append(vc)
                val_accuracy_pix.append(va_pix)
                val_accuracy_iou.append(va_iou)
                val_accuracy_per_class.append(va_class)

                # Print information
                print("\nEpoch: " + str(epoch))
                print('\ntrain cost: ' + str(tc))
                print('train pixel accuracy: {:.2f}%'.format(ta_pix * 100))
                print('train mean iou accuracy: {:.2f}%'.format(ta_iou * 100))
                print('train mean per class accuracy: {:.2f}% \n'.format(ta_class * 100))
                print('val cost: ' + str(vc))
                print('val pixel accuracy: {:.2f}%'.format(va_pix * 100))
                print('val mean iou accuracy: {:.2f}%'.format(va_iou * 100))
                print('val mean per class accuracy: {:.2f}% \n'.format(va_class * 100))

                # Save model if specified
                if epochs > OPTIMIZATION_EPOCHS and SAVE_MODEL == 1:
                    save_path = saver.save(sess, SAVE_PATH + "fcn32_model.ckpt", global_step=epoch)
                    print("Epoch %d - Model saved in path: %s \n" % (epoch, save_path))
                    sys.stdout.flush()

                # Check if loss is lower than in previous epochs
                if vc <= best_val_loss:
                    lowest_val_loss_epoch = epoch
                    patience = 0
                else:
                    patience += 1
                    if patience == 10:
                        print('End of training due to early stopping! \n')
                        break

                # Increase epoch and decay learning rate
                epoch += 1
                if epoch % 50 == 0:
                    learning_rate = lr_decay(learning_rate)

                # Check end time and compute epoch time lapse
                end = datetime.now()
                delta = end - start
                print('\tEpoch trained in %d hours, %d minutes and %d seconds' % (
                    delta.seconds // 3600, ((delta.seconds // 60) % 60), (delta.seconds) % 60))
                sys.stdout.flush()

        except tf.errors.OutOfRangeError:
            print('End of training! \n')

        # Save accuracy and cost
        if epochs > OPTIMIZATION_EPOCHS and SAVE_MODEL == 1:
            np.save('fcn32_train_loss.npy', train_cost)
            np.save('fcn32_val_loss.npy', val_cost)
            np.save('fcn32_train_accuracy_pix.npy', train_accuracy_pix)
            np.save('fcn32_val_accuracy_pix.npy', val_accuracy_pix)
            np.save('fcn32_train_accuracy_iou.npy', train_accuracy_iou)
            np.save('fcn32_val_accuracy_iou.npy', val_accuracy_iou)
            np.save('fcn32_train_accuracy_class.npy', train_accuracy_per_class)
            np.save('fcn32_val_accuracy_class.npy', val_accuracy_per_class)

        # Close summary writer
        if DEBUG:
            self.summary_writer.close()

        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
        sess.close()
        tf.reset_default_graph()

        print('learningRate: ', initial_learning_rate, 'Dropout: ', dropout)
        print('Loss function: ', np.min(val_cost))
        print('----------------- \n')
        sys.stdout.flush()

        return (np.min(val_cost))

    def predict(self):

        # Initialization
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess = tf.Session()
        sess.run(init_op)

        # Create a saver
        saver = tf.train.Saver()
        saver.restore(sess, RESTORE_PATH)
        print('Load checkpoint %s' % RESTORE_PATH)
        print("Model restored.")

if __name__ == "__main__":
    # 1. Imports
    import tensorflow as tf
    import numpy as np

    def execute(args):
        tf.reset_default_graph()
        model = deepLabNet()
        output = model.train(args)
        tf.reset_default_graph()
        return output

    # define a search space

    space = hp.choice('experiment number',
                      [
                          (hp.uniform('learning_rate', 0.01, 0.0001),
                           hp.uniform('dropout_prob', 0.5, 0.8),
                           hp.uniform('Epochs', OPTIMIZATION_EPOCHS, OPTIMIZATION_EPOCHS))
                      ])

    best = fmin(execute, space, algo=tpe.suggest, max_evals=EVALUATIONS)

    print('Best learningRate: ', best['learning_rate'], 'Best Dropout: ', best['dropout_prob'])
    print('-----------------\n')
    print('Starting training with optimized hyperparameters... \n')

    execute((best['learning_rate'], best['dropout_prob'], 2))

    # Check that prediction works
    # tf.reset_default_graph()
    # model = deepLabNet()
    # model.build_train_op()
    # model.predict()