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

        self.X = tf.placeholder(tf.float32, shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        self.y = tf.placeholder(tf.int32, shape=(None, SEGMENTATION_HEIGHT, SEGMENTATION_WIDTH, 1))
        self.lr = tf.placeholder(tf.float32)
        self.pkeep = tf.placeholder(tf.float32)
        self.is_train = tf.placeholder(tf.bool, name="is_train")

    # BUILD UP THE MODEL ***********************************************************************************************

    def build_model(self): # ResNet 18
        print('\nBuilding model')

        # filters = [128, 128, 256, 512, 1024]
        filters = [64, 64, 128, 256, 512]
        kernels = [7, 3, 3, 3, 3]
        strides = [2, 0, 2, 2, 2]

        # conv1
        print('\tBuilding unit: conv1')
        with tf.variable_scope('conv1'):
            x = myconv(self.X, kernels[0], filters[0], strides[0], name="conv")
            x = mybn(x, self.is_train, name="bn")
            x = myrelu(x)
            self.shape_conv1 = tf.shape(x)
            x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

        # conv2_x
        x = residual_block(x,kernels[1], self.is_train, name='conv2_1')
        x = residual_block(x, kernels[1], self.is_train, name='conv2_2')
        self.shape_conv2 = tf.shape(x)

        # conv3_x
        x = first_residual_block(x, kernels[2], filters[2], strides[2], self.is_train, name='conv3_1')
        x = residual_block(x, kernels[2], self.is_train, name='conv3_2')
        self.shape_conv3 = tf.shape(x)

        # conv4_x
        x = first_residual_block(x, kernels[3], filters[3], strides[3], self.is_train, name='conv4_1')
        x = residual_block(x, kernels[3], self.is_train, name='conv4_2')
        self.shape_conv4 = tf.shape(x)

        # conv5_x
        x = first_residual_atrous_block(x, kernels[4], filters[4], strides[4], self.is_train, name='conv5_1')
        x = residual_atrous_block(x, kernels[4], self.is_train, name='conv5_2')
        self.shape_conv5 = tf.shape(x)

        # aspp
        x = atrous_spatial_pyramid_pooling_block(x, self.is_train, depth=256, name = 'aspp_1')
        self.shape_aspp = tf.shape(x)

        print('\tBuilding unit: class scores') # Maybe another layer ???
        x = myconv(x, 1, CLASSES, 1, name="class_scores")
        self.shape_class_scores = tf.shape(x)

        # upsample logits
        print('\tBuilding unit: upsample')
        x = tf.image.resize_images(
            x,
            tf.stack([SEGMENTATION_HEIGHT, SEGMENTATION_WIDTH]),
            method=tf.image.ResizeMethod.BILINEAR,
            align_corners=False)
        logits = x
        self.shape_upsampled_logits = tf.shape(logits)

        logits_by_num_classes = tf.reshape(logits, [-1, CLASSES])

        # Probs and Predictions
        probs = tf.nn.softmax(logits, name='softmax_tensor')
        probs_by_num_classes = tf.reshape(probs, [-1, CLASSES])
        preds = tf.argmax(logits, axis=3, output_type=tf.int32)
        preds_flat = tf.reshape(preds, [-1, ])
        labels_flat = tf.reshape(self.y, [-1, ])

        # Remove non valid indices
        valid_indices = tf.multiply(tf.to_int32(labels_flat <= CLASSES - 1), tf.to_int32(labels_flat > -1))
        valid_probs = tf.dynamic_partition(probs_by_num_classes, valid_indices, num_partitions=2)[1]
        valid_logits = tf.dynamic_partition(logits_by_num_classes, valid_indices, num_partitions=2)[1]
        valid_labels = tf.dynamic_partition(labels_flat, valid_indices, num_partitions=2)[1]
        valid_preds = tf.dynamic_partition(preds_flat, valid_indices, num_partitions=2)[1]
        summary_histogram(valid_preds, name="valid_preds")

        # ACCURACY *****************************************************************************************************

        self.pixel_acc, self.mean_iou = compute_accuracy(valid_preds, valid_labels)
        summary_scalar(self.pixel_acc, name="pixel_accuracy")
        summary_scalar(self.mean_iou, name="mean_iou")

        # COST *********************************************************************************************************

        self.cross_entropy = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=valid_logits,
                                                                              labels=valid_labels,
                                                                              name="entropy")))
        sys.stdout.flush()
        return valid_logits, valid_preds, self.cross_entropy, self.pixel_acc, self.mean_iou

    # BUILD TRAIN OPERATION ********************************************************************************************

    def build_train_op(self):

        # Build Model
        logits, preds, loss, pixel_acc, mean_iou = self.build_model()

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
        self.train_op = tf.group(*(update_ops + [apply_grad_op])) # runs all operarions

        print("Setting up summary op")
        self.summary_op = tf.summary.merge_all()

    # TRAIN EPOCH ******************************************************************************************************

    def train_epoch(self, sess, train_images, train_labels, current_learning_rate, dropout):

        train_cost = []
        train_accuracy_miou = []
        train_accuracy_pixel = []

        steps = int(float(TRAIN_IMAGES) * (float(CROPS_PER_IMAGE) / float(BATCH_SIZE)))

        for i in range(steps):
            batch_X, batch_Y = sess.run([train_images, train_labels])
            batch_X, batch_Y = preprocess_data(batch_X, batch_Y)
            feed_dict = {self.X: batch_X, self.y: batch_Y, self.pkeep: dropout, self.lr: current_learning_rate, self.is_train: 1}
            sess.run(self.train_op, feed_dict=feed_dict, options=run_options)
            tc, ta_pix, ta_iou, summary_str = sess.run([self.cross_entropy, self.pixel_acc, self.mean_iou, self.summary_op],
                                                       feed_dict=feed_dict,
                                                       options=run_options)
            if DEBUG == 1:
                self.summary_writer.add_summary(summary_str)
            train_cost.append(tc)
            train_accuracy_pixel.append(ta_pix)
            train_accuracy_miou.append(ta_iou)
            if i % 100 == 0:
                print('batch: ' + str(i))
                sys.stdout.flush()

        min_position = np.argmin(train_cost)
        tc_min = train_cost[min_position]
        ta_pix_min = train_accuracy_pixel[min_position]
        ta_iou_min = train_accuracy_miou[min_position]

        return tc_min, ta_pix_min, ta_iou_min

    # GET VAL COST/ACC *************************************************************************************************

    def evaluate_model_on_test_set(self, sess, test_images, test_labels):

        test_cost = []
        test_accuracy_miou = []
        test_accuracy_pixel = []

        steps = int(float(TEST_IMAGES) * (float(CROPS_PER_IMAGE) / float(BATCH_SIZE)))

        for i in range(steps):
            batch_X, batch_Y = sess.run([test_images, test_labels], options=run_options)
            batch_X, batch_Y = preprocess_data(batch_X, batch_Y)
            feed_dict = {self.X: batch_X, self.y: batch_Y, self.pkeep: 1, self.is_train: 1}
            te_c, te_a_pix, te_a_iou = sess.run([self.cross_entropy, self.pixel_acc, self.mean_iou], feed_dict=feed_dict, options=run_options)
            test_cost.append(te_c)
            test_accuracy_pixel.append(te_a_pix)
            test_accuracy_miou.append(te_a_iou)
            if i % 100 == 0:
                print('batch: ' + str(i))
                sys.stdout.flush()

        te_c_avg = np.mean(test_cost)
        te_a_pix_avg = np.mean(test_accuracy_pixel)
        te_a_iou_avg = np.mean(test_accuracy_miou)

        return te_c_avg, te_a_pix_avg, te_a_iou_avg

    # CHECK NETWORK DIMs ***********************************************************************************************

    def checkDimensions(self):
        # Initialization
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess = tf.Session()
        sess.run(init_op)

        # Toy batch
        batch_X = np.ones((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        batch_Y = np.ones((1, SEGMENTATION_HEIGHT, SEGMENTATION_WIDTH, 1))

        print('\nNetwork dimensions:')
        shape = batch_X.shape
        print('\tInput: [x, %d %d %d]' % (shape[1],shape[2],shape[3]))
        shape = np.array(
            sess.run(self.shape_conv1, feed_dict={self.X: batch_X, self.y: batch_Y, self.is_train: 1, self.pkeep: 1}))
        print('\tconv 1: [x, %d %d %d]' % (shape[1],shape[2],shape[3]))
        shape = np.array(
            sess.run(self.shape_conv2, feed_dict={self.X: batch_X, self.y: batch_Y, self.is_train: 1, self.pkeep: 1}))
        print('\tconv 2_x: [x, %d %d %d]' % (shape[1], shape[2], shape[3]))
        shape = np.array(
            sess.run(self.shape_conv3, feed_dict={self.X: batch_X, self.y: batch_Y, self.is_train: 1, self.pkeep: 1}))
        print('\tconv 3_x: [x, %d %d %d]' % (shape[1], shape[2], shape[3]))
        shape = np.array(
            sess.run(self.shape_conv4, feed_dict={self.X: batch_X, self.y: batch_Y, self.is_train: 1, self.pkeep: 1}))
        print('\tconv 4_x: [x, %d %d %d]' % (shape[1], shape[2], shape[3]))
        shape = np.array(
            sess.run(self.shape_conv5, feed_dict={self.X: batch_X, self.y: batch_Y, self.is_train: 1, self.pkeep: 1}))
        print('\tconv 5_x: [x, %d %d %d]' % (shape[1], shape[2], shape[3]))
        shape = np.array(
            sess.run(self.shape_aspp, feed_dict={self.X: batch_X, self.y: batch_Y, self.is_train: 1, self.pkeep: 1}))
        print('\taspp: [x, %d %d %d]' % (shape[1], shape[2], shape[3]))
        shape = np.array(
            sess.run(self.shape_class_scores,
                     feed_dict={self.X: batch_X, self.y: batch_Y, self.is_train: 1, self.pkeep: 1}))
        print('\tpreds: [x, %d %d %d]' % (shape[1], shape[2], shape[3]))
        shape = np.array(
            sess.run(self.shape_upsampled_logits, feed_dict={self.X: batch_X, self.y: batch_Y, self.is_train: 1, self.pkeep: 1}))
        print('\tupsampled logits: [x, %d %d %d]' % (shape[1], shape[2], shape[3]))
        shape = np.array(
            sess.run(self.shape_probs,
                     feed_dict={self.X: batch_X, self.y: batch_Y, self.is_train: 1, self.pkeep: 1}))
        print('\tprobs: [x, %d %d %d]' % (shape[1], shape[2], shape[3]))
        shape = np.array(
            sess.run(self.shape_preds,
                     feed_dict={self.X: batch_X, self.y: batch_Y, self.is_train: 1, self.pkeep: 1}))
        print('\tpreds: [x, %d %d %d]' % (shape[1], shape[2], shape[3]))

    # TRAIN MODEL ******************************************************************************************************

    def train(self, args):

        learning_rate, dropout, epochs = args
        initial_learning_rate = learning_rate

        # Get images and labels
        with tf.device('/cpu:0'):
            train_images, train_labels = read_and_decode(DATASET_PATH + 'train_dataset.tfrecords', epochs, BATCH_SIZE)
            test_images, test_labels = read_and_decode(DATASET_PATH + 'test_dataset.tfrecords', epochs, BATCH_SIZE)

        # Initialization
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess = tf.Session()
        sess.run(init_op)

        # Load Imagenet Pretrained Weights
        print("\nLoading ResNet18 pretrained weights")
        resnet18_weights = np.load("resnet18_weights.npy")
        all_vars = tf.trainable_variables()
        notloaded = []
        print('Successful:')
        for v in all_vars:
            try:
                assign_op = v.assign(resnet18_weights.item().get(v.op.name))
                sess.run(assign_op)
                print('\t' + v.op.name)
            except Exception, e:
                notloaded.append(v.op.name)
        print('Unsuccessful:')
        for name in notloaded:
            print('\t'+name)
        sys.stdout.flush()

        # Create a saver
        saver = tf.train.Saver(max_to_keep=5)
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
        val_accuracy_pix = []
        val_accuracy_iou = []

        print("Starting to train...")
        sys.stdout.flush()

        try:
            best_test_cost = float('Inf')
            epoch = 1
            while (True):

                # Check start time
                start = datetime.now()

                # Train for 1 epoch on training set
                tc, ta_pix, ta_iou = self.train_epoch(sess, train_images, train_labels, learning_rate, dropout)
                train_cost.append(tc)
                train_accuracy_pix.append(ta_pix)
                train_accuracy_iou.append(ta_iou)

                # Check on test/val dataset
                vc, va_pix, va_iou = self.evaluate_model_on_test_set(sess, test_images, test_labels)
                val_cost.append(vc)
                val_accuracy_pix.append(va_pix)
                val_accuracy_iou.append(va_iou)

                # Print epoch information
                print("\nEpoch: " + str(epoch))
                print('\ttrain cost: ' + str(tc))
                print('\ttrain pixel accuracy: {:.2f}%'.format(ta_pix * 100))
                print('\ttrain mean iou accuracy: {:.2f}%'.format(ta_iou * 100))
                print('\ttest cost: ' + str(vc))
                print('\ttest pixel accuracy: {:.2f}%'.format(va_pix * 100))
                print('\ttest mean iou accuracy: {:.2f}%'.format(va_iou * 100))
                sys.stdout.flush()

                # Save model if specified
                if epochs > OPTIMIZATION_EPOCHS and SAVE_MODEL == 1 and vc < best_test_cost and np.isfinite(vc):
                    best_test_cost = vc
                    save_path = saver.save(sess, SAVE_PATH + "deepLabv3_model.ckpt")
                    print("\tEpoch %d - Model saved in path: %s" % (epoch, save_path))
                    sys.stdout.flush()

                # Increase epoch and decay learning rate
                epoch += 1
                if epoch % 50 == 0:
                    learning_rate = lr_decay(learning_rate)

                # Check end time and compute epoch time lapse
                end = datetime.now()
                delta = end - start
                print('\tEpoch trained in %d hours, %d minutes and %d seconds\n' % (
                    delta.seconds // 3600, ((delta.seconds // 60) % 60), (delta.seconds) % 60))
                sys.stdout.flush()

        except tf.errors.OutOfRangeError:
            print('End of training! \n')
            sys.stdout.flush()

        # Save accuracy and cost
        #if epochs > OPTIMIZATION_EPOCHS and SAVE_MODEL == 1:
        #    np.save('fcn32_Train_Loss.npy', train_cost)
        #    np.save('fcn32_Validation_Loss.npy', val_cost)
        #    np.save('fcn32_Train_Accuracy_Pix.npy', train_accuracy_pix)
        #    np.save('fcn32_Validation_Accuracy_Pix.npy', val_accuracy_pix)
        #    np.save('fcn32_Train_Accuracy_Iou.npy', train_accuracy_iou)
        #    np.save('fcn32_Validation_Accuracy_Iou.npy', val_accuracy_iou)

        # Close summary writer
        if DEBUG == 1:
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
        model.build_train_op()
        output = model.train(args)
        tf.reset_default_graph()
        return output


    execute((0.001, 0.55, 200))

    # define a search space

    #space = hp.choice('experiment number',
    #                  [
    #                      (hp.uniform('learning_rate', 0.001, 0.00001),
    #                       hp.uniform('dropout_prob', 0.5, 0.8),
    #                       hp.uniform('Epochs', OPTIMIZATION_EPOCHS, OPTIMIZATION_EPOCHS))
    #                  ])

    #best = fmin(execute, space, algo=tpe.suggest, max_evals=EVALUATIONS)

    #print('Best learningRate: ', best['learning_rate'], 'Best Dropout: ', best['dropout_prob'])
    #print('-----------------\n')
    #print('Starting training with optimized hyperparameters... \n')

    #execute((best['learning_rate'], best['dropout_prob'], 2))

    # Check that prediction works
    # tf.reset_default_graph()
    # model = deepLabNet()
    # model.build_train_op()
    # model.predict()