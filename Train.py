# IMPORTS
from GlobalVariables import*
from datetime import datetime
from hyperopt import hp, fmin, tpe
import sys
import os

model = 'resnet18'
#path = '/DEEPLAB'
path = '/Users/albertbou/PycharmProjects/DeepLab_v3'
sys.path.append( path + '/models')
if model == 'resnet18':
    from Utils_18 import *
    import ResNet18_deeplab
elif model == 'resnet101':
    from Utils_101 import*
    import ResNet101_deeplab
else:
    print('Specified model is no valid')
    sys.exit(1)

# NETWORK CLASS

class deepLabNet:

    # DEFINE CLASS *****************************************************************************************************

    def __init__(self):

        self.lr = tf.placeholder(tf.float32)
        self.weight_decay = tf.placeholder(tf.float32)

    # BUILD TRAIN OPERATION ********************************************************************************************

    def build_train_op(self, X, y, is_train, pkeep):

        # Build Model
        if model == 'resnet18':
            logits, preds, loss, pixel_acc, mean_iou, per_class_acc = ResNet18_deeplab.deeplab_rn18(X, y, is_train, pkeep)
        elif model == 'resnet101':
            logits, preds, loss, pixel_acc, mean_iou, per_class_acc = ResNet101_deeplab.deeplab_rn101(X, y, is_train, pkeep)

        # Learning rate - save in summary
        tf.summary.scalar('learing_rate', self.lr)

        # Train step
        opt = tf.train.MomentumOptimizer(self.lr, MOMENTUM)

        # Add l2 loss
        costs = [tf.nn.l2_loss(var) for var in tf.get_collection(WEIGHT_DECAY_KEY)]
        l2_loss = tf.multiply(self.weight_decay, tf.add_n(costs))
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
        train_op = tf.group(*(update_ops + [apply_grad_op])) # runs all operations

        print("Setting up summary op")
        summary_op = tf.summary.merge_all()

        return total_loss, pixel_acc, mean_iou, per_class_acc, train_op, summary_op

    # TRAIN EPOCH ******************************************************************************************************

    def train_epoch(self, sess, current_learning_rate):

        sess.run(self.reset_op)
        epoch_loss = 0
        steps = int(float(TRAIN_IMAGES) * (float(CROPS_PER_IMAGE) / float(BATCH_SIZE)))
        feed_dict = {self.lr: current_learning_rate, self.weight_decay: self.w_decay}

        for i in range(steps):
            tr_loss, tr_pixel_acc, tr_mean_iou_acc, tr_mean_per_class_acc, _ = sess.run(
                [self.train_loss, self.train_pixel_acc, self.train_mean_iou_acc, self.train_mean_per_class_acc, self.train_step],
                feed_dict=feed_dict)
            if np.isnan(tr_loss):
                #print('got NaN as the loss value for 1 image')
                epoch_loss += float('inf')
            else:
                epoch_loss += tr_loss

            if DEBUG:
                summary_str = sess.run([self.summary_op], feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_str)

        epoch_loss = epoch_loss / steps
        return epoch_loss, tr_pixel_acc, tr_mean_iou_acc, tr_mean_per_class_acc

    # GET VAL COST/ACC *************************************************************************************************

    def evaluate_model_on_val_set(self, sess):

        sess.run(self.reset_op)
        epoch_loss = 0
        steps = int(float(VALIDATION_IMAGES) * (float(CROPS_PER_IMAGE) / float(BATCH_SIZE)))
        feed_dict = {self.weight_decay: 0.0}

        for i in range(steps):
            v_loss, v_pixel_acc, v_mean_iou_acc, v_mean_per_class_acc = \
                sess.run([self.val_loss, self.val_pixel_acc, self.val_mean_iou_acc, self.val_mean_per_class_acc],
                         feed_dict = feed_dict)
            if np.isnan(v_loss):
                #print('got NaN as the loss value for 1 image')
                epoch_loss += float('inf')
            else:
                epoch_loss += v_loss

        epoch_loss = epoch_loss / steps
        return epoch_loss, v_pixel_acc, v_mean_iou_acc, v_mean_per_class_acc

    # TRAIN MODEL ******************************************************************************************************

    def train(self, args):

        learning_rate, dropout, self.w_decay, epochs = args
        initial_learning_rate = learning_rate
        epochs = int(epochs)

        # Get images and labels
        with tf.device('/cpu:0'):
            train_images, train_labels = read_and_decode(DATASET_PATH + 'train_dataset.tfrecords', epochs, BATCH_SIZE)
            val_images, val_labels = read_and_decode_val_and_test(DATASET_PATH + 'validation_dataset.tfrecords', epochs, BATCH_SIZE)

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
        print("\nLoading ResNet pretrained weights")

        if model == "resnet18":
            resnet_weights = np.load("resnet18_weights.npy")
        elif model == "resnet101":
            resnet_weights = np.load("resnet101_weights.npy")

        all_vars = tf.trainable_variables()
        notloaded = []
        print('Successful:')
        for v in all_vars:
            try:
                if model == "resnet18":
                    assign_op = v.assign(resnet_weights.item().get(v.op.name[6:]))
                elif model == "resnet101":
                    assign_op = v.assign(resnet_weights.item().get(v.op.name[:]))
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
                ta_class = np.mean(ta_class)
                train_cost.append(tc)
                train_accuracy_pix.append(ta_pix)
                train_accuracy_iou.append(ta_iou)
                train_accuracy_per_class.append(ta_class)

                # Compute cost/accuracy on val dataset
                vc, va_pix, va_iou, va_class = self.evaluate_model_on_val_set(sess)
                va_class = np.mean(va_class)
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

                # Check if loss is lower than in previous epochs
                if vc <= best_val_loss:

                    # Save model if specified
                    if epochs > OPTIMIZATION_EPOCHS and SAVE_MODEL == 1:
                        save_path = saver.save(sess, SAVE_PATH + "deeplab_model.ckpt", global_step=epoch)
                        print("Epoch %d - Model saved in path: %s \n" % (epoch, save_path))
                        sys.stdout.flush()

                    best_val_loss = vc
                    lowest_val_loss_epoch = epoch
                    patience = 0
                else:
                    patience += 1
                    # if 5 epochs without any improvement, decay learning rate
                    if patience % 5 == 0:
                        print('\nLearning rate decay\n')
                        learning_rate = lr_decay(learning_rate)
                    # if 15 epochs without any improvement, early stopping
                    if patience == 30:
                        print('End of training due to early stopping! \n')
                        break

                # Increase epoch and decay learning rate
                epoch += 1

                # Check end time and compute epoch time lapse
                end = datetime.now()
                delta = end - start
                print('\tEpoch trained in %d hours, %d minutes and %d seconds' % (
                    delta.seconds // 3600, ((delta.seconds // 60) % 60), (delta.seconds) % 60))
                sys.stdout.flush()

        except tf.errors.OutOfRangeError:
            print('\nEnd of training! \n')

        # Save accuracy and cost
        if epochs > OPTIMIZATION_EPOCHS and SAVE_MODEL == 1:
            np.save('deeplab_train_loss.npy', train_cost)
            np.save('deeplab_val_loss.npy', val_cost)
            np.save('deeplab_train_accuracy_pix.npy', train_accuracy_pix)
            np.save('deeplab_val_accuracy_pix.npy', val_accuracy_pix)
            np.save('deeplab_train_accuracy_iou.npy', train_accuracy_iou)
            np.save('deeplab_val_accuracy_iou.npy', val_accuracy_iou)
            np.save('deeplab_train_accuracy_class.npy', train_accuracy_per_class)
            np.save('deeplab_val_accuracy_class.npy', val_accuracy_per_class)

        # Close summary writer
        if DEBUG:
            self.summary_writer.close()

        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
        sess.close()
        tf.reset_default_graph()

        print('Learning Rate: ', initial_learning_rate, 'Dropout: ', dropout, 'Weight Decay', self.weight_decay)
        print('Loss function: ', np.min(val_cost))
        print('\nBest model obtained in epoch ' + str(lowest_val_loss_epoch) + '\n')
        print('----------------- \n')
        sys.stdout.flush()

        return (np.min(val_cost))

if __name__ == "__main__":
    # 1. Imports
    import tensorflow as tf
    import numpy as np

    def deeplab(args):
        tf.reset_default_graph()
        model = deepLabNet()
        output = model.train(args)
        tf.reset_default_graph()
        return output


    search = "grid"  # "grid" or "random"

    if search == "random":

        # define a search space
        space = hp.choice('experiment number',
                          [
                              (hp.uniform('learning_rate', 0.0001, 0.01),
                               hp.uniform('dropout_prob', 0.5, 1.0),
                               hp.uniform('weight_decay', 1.0e-6, 1.0e-4),
                               hp.quniform('Epochs', OPTIMIZATION_EPOCHS, OPTIMIZATION_EPOCHS + 1, OPTIMIZATION_EPOCHS))
                          ])

        best = fmin(deeplab, space, algo=tpe.suggest, max_evals=EVALUATIONS)

        print('Best learningRate: ', best['learning_rate'], 'Best Dropout: ', best['dropout_prob'], 'Best weight decay',
              best['weight_decay'])
        print('-----------------\n')
        print('Starting training with optimized hyperparameters... \n')
        sys.stdout.flush()

        deeplab((best['learning_rate'], best['dropout_prob'], best['weight_decay'], EPOCHS))

    elif search == "grid":
        min_loss = float('inf')
        learning_rate = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6]
        dropout_prob = [1.0, 0.75, 0.5]
        weight_decay = [0.0, 1.0e-4, 1.0e-6]
        for lr in learning_rate:
            for dr in dropout_prob:
                for wd in weight_decay:
                    loss = deeplab((lr, dr, wd, OPTIMIZATION_EPOCHS))
                    if (loss < min_loss):
                        best_lr = lr
                        best_dr = dr
                        best_wd = wd
                        min_loss = loss
        deeplab((best_lr, best_dr, best_wd, EPOCHS))