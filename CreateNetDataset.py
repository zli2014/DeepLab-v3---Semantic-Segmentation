
# PARAMETERS ***********************************************************************************************************

number_of_crops = 4

# IMPORTS **************************************************************************************************************

from random import shuffle
import glob
import cv2
import tensorflow as tf
import numpy as np
from GlobalVariables import*
import sys
from PIL import Image
import math
import pdb

# FUNCTIONS ************************************************************************************************************

def extract_patch_from_image(img,patch):

    """this function extracts the specified patch from the original image"""

    # Read an image and resize to (IMAGE_HEIGHT, IMAGE_WIDTH)
    img = img[patch[0]:patch[0]+IMAGE_HEIGHT,patch[1]:patch[1]+IMAGE_WIDTH,:]
    # Cast to float32
    img = img.astype(np.float32)
    return img

def extract_patch_from_label(label,patch):

    """this function extracts the specified patch from the original label image"""

    # Read an image and resize to (IMAGE_HEIGHT, IMAGE_WIDTH)
    label = label[patch[0]:patch[0]+IMAGE_HEIGHT,patch[1]:patch[1]+IMAGE_WIDTH]
    # Cast to float32
    label = label.astype(np.float32)
    return label

def prepare_and_write_image_patches(writer,image_addrs,stuff_addrs,i):

    """this functions creates patches from an image/label pair and writes thes in a tfrecord file"""

    # Depending on the specified number of crops per image, find indices in x and y axis
    if number_of_crops%2 == 0:
        xindex = number_of_crops/2
        yindex = number_of_crops/2
    else:
        xindex = number_of_crops/2 + 1
        yindex = number_of_crops/2

    # the number of images != number of labels!
    #image_address = '/data/train2017/'+stuff_addrs[i][-16:-3]+'jpg'
    image_address = '/data/val2017/'+stuff_addrs[i][-16:-3]+'jpg'

    # Load the image
    batch_mean = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH,3))
    img = np.array(cv2.imread(image_address))
    label = np.array(Image.open(stuff_addrs[i]))

    # Check that image and label dimensions match
    if img.shape[0:2] != label.shape:
        print("Error: detected image/label pair not matching in size")
        sys.exit(1)

    # Check that the image under consideration fullfils minimum height/width requirements
    height = img.shape[0]
    width = img.shape[1]
    if height<(IMAGE_HEIGHT+4) or width <(IMAGE_WIDTH+4):
        return -1

    # Define x and y gaps between crops
    xstep = int(math.floor((height-IMAGE_HEIGHT)/xindex)-1)
    ystep = int(math.floor((width-IMAGE_WIDTH)/yindex)-1)

    # Generate a list with the desired patches (top left cornet position)
    patches = []
    x = 0
    y = 0
    xposition = []
    yposition = []
    for i in range (xindex):
        xposition.append(x)
        x += xstep
    for i in range (yindex):
        yposition.append(y)
        y += ystep
    for i in range (xindex):
        for j in range (yindex):
            patches.append([xposition[i],yposition[j]])

    # Load the patches and return the batch_mean
    loaded_patches = 0
    for i in range(number_of_crops):

        # Extract patches
        patch_img = extract_patch_from_image(img,patches[i])
        patch_label = extract_patch_from_label(label,patches[i])
        batch_mean = np.add(batch_mean, patch_img)

        #print(loaded_patches)
        loaded_patches += 1

        # Create a feature
        feature = {'image': _bytes_feature(tf.compat.as_bytes(patch_img.tostring())),
                   'label': _bytes_feature(tf.compat.as_bytes(patch_label.tostring()))}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    # Return batch mean
    return batch_mean

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

if __name__ == "__main__":

    """Creates network datasets"""

    try:
        arg1 = sys.argv[1]
    except IndexError:
        print("Command line argument missing. Usage: CreateNetDataset.py <mode>")
        sys.exit(1)

    save_mean = 0

    if arg1 == 'train':
        images_path = ORIGINAL_DATASET_PATH + '/train2017/*.jpg'
        stuff_path = ORIGINAL_DATASET_PATH + '/stuff_train2017_pixelmaps/*.png'
        dataset_name = 'train_dataset.tfrecords'
        save_mean = 1

    elif arg1 == 'test':
        images_path = ORIGINAL_DATASET_PATH + '/val2017/*.jpg'
        stuff_path = ORIGINAL_DATASET_PATH + '/stuff_val2017_pixelmaps/*.png'
        dataset_name = 'test_dataset.tfrecords'

    elif arg1 == 'validation':
        images_path = ORIGINAL_DATASET_PATH + '/val2017/*.jpg'
        stuff_path = ORIGINAL_DATASET_PATH + '/stuff_val2017/*.png'
        dataset_name = 'val_dataset.tfrecords'

    # Read addresses and labels from the 'train' folder
    image_addrs = glob.glob(images_path)
    stuff_addrs = glob.glob(stuff_path)

    # Sort the list of addresses
    train_image_addrs = sorted(image_addrs)
    train_stuff_addrs = sorted(stuff_addrs)

    # Check that train_image_addrs and train_stuff_addrs have the same length
    if len(train_image_addrs) != len(train_stuff_addrs):
        print("Error: image address list length and label address list length are different")
    #    sys.exit(1)

    # Specify address to save the TFRecords file
    train_filename = dataset_name

    # Open the TFRecords file
    writer = tf.python_io.TFRecordWriter(train_filename)

    # Define dataset mean
    mean = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH,3))
    images = len(train_stuff_addrs)

    # Define total images in the dataset and total ejected images
    total_images = 0
    total_rejected = 0

    for i in range(images):
        # Print how many images are saved every 1000 images
        if not i % 1000:
            print 'Created data: {}/{}'.format(i, images)
        batch_mean = prepare_and_write_image_patches(writer,train_image_addrs,train_stuff_addrs,i)
        if batch_mean is not -1:
            mean = np.add(mean, batch_mean)
            total_images += 1
        else:
            total_rejected += 1

    # Normalize mean
    mean = mean/(images*CROPS_PER_IMAGE)

    # Print some info
    print('Dataset creation summary:')
    print('\tTotal number of images in the directory: '+str(images))
    print('\tTotal number of images rejected due to their dimensions: '+str(total_images))
    print('\tTotal number of images in the dataset is: '+str(total_rejected))

    # Save mean
    if save_mean == 1:
        np.save('train_mean',mean)

    writer.close()
    sys.stdout.flush()

