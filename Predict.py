
path = '/Users/albertbou/PycharmProjects/FCN'
model = 'fcn32'
model_path = ''

# Imports
from Utils_18 import *
import sys

model = 'resnet18'
#path = '/Users/albertbou/PycharmProjects/DeepLab_v3'
#path = '/DEEPLAB'
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

print("\nReading test data")
test_images, test_labels = read_and_decode_val_and_test(DATASET_PATH + 'test_dataset.tfrecords', epochs=1, batch_size=1)

# Define test model
# Build Model
if model == 'resnet18':
    _, _, test_loss, test_pixel_acc, test_mean_iou_acc, test_mean_per_class_acc =\
        ResNet18_deeplab.deeplab_rn18(X=test_images, y=test_labels, is_train = tf.constant(False))
elif model == 'resnet101':
    _, _, test_loss, test_pixel_acc, test_mean_iou_acc, test_mean_per_class_acc =\
        ResNet101_deeplab.deeplab_rn101(X=test_images, y=test_labels, is_train = tf.constant(False))

# Initialization
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess = tf.Session(config=config)
sess.run(init_op)

# Restore model
restore_saver = tf.train.Saver()
restore_saver.restore(sess, model_path)
print('\nLoading model from specified checkpoint...\n')

loss = 0
steps = TEST_IMAGES

for i in range(steps):
    t_loss, t_pixel_acc, t_mean_iou_acc, t_mean_per_class_acc = \
        sess.run([test_loss, test_pixel_acc, test_mean_iou_acc, test_mean_per_class_acc])
    if np.isnan(t_loss):
        # print('got NaN as the loss value for 1 image')
        loss += float('inf')
    else:
        loss += t_loss
loss = loss / steps

t_mean_per_class_acc = np.mean(t_mean_per_class_acc)

# Print information
print('test cost: ' + str(loss))
print('test pixel accuracy: {:.2f}%'.format(t_pixel_acc * 100))
print('test mean iou accuracy: {:.2f}%'.format(t_mean_iou_acc * 100))
print('test mean per class accuracy: {:.2f}% \n'.format(t_mean_per_class_acc * 100))



