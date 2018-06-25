
# Download the ResNet-18 torch checkpoint
# wget https://d2j0dndfm35trm.cloudfront.net/resnet-18.t7
# Download the ResNet-101 .npy weights
# http://elbereth.zemris.fer.hr/kivan/resnet/

# Imports
import numpy as np
import torchfile

def import_resnet18_weights():

    # Path
    T7_PATH = './resnet-18.t7'

    # Open ResNet-18 torch checkpoint
    print('Open ResNet-18 torch checkpoint: %s' % T7_PATH)
    o = torchfile.load(T7_PATH)

    # Load weights in a brute-force way
    print('Load weights in a brute-force way')
    conv1_weights = o.modules[0].weight
    conv1_bn_gamma = o.modules[1].weight
    conv1_bn_beta = o.modules[1].bias
    conv1_bn_mean = o.modules[1].running_mean
    conv1_bn_var = o.modules[1].running_var

    conv2_1_weights_1  = o.modules[4].modules[0].modules[0].modules[0].modules[0].weight
    conv2_1_bn_1_gamma = o.modules[4].modules[0].modules[0].modules[0].modules[1].weight
    conv2_1_bn_1_beta  = o.modules[4].modules[0].modules[0].modules[0].modules[1].bias
    conv2_1_bn_1_mean  = o.modules[4].modules[0].modules[0].modules[0].modules[1].running_mean
    conv2_1_bn_1_var   = o.modules[4].modules[0].modules[0].modules[0].modules[1].running_var
    conv2_1_weights_2  = o.modules[4].modules[0].modules[0].modules[0].modules[3].weight
    conv2_1_bn_2_gamma = o.modules[4].modules[0].modules[0].modules[0].modules[4].weight
    conv2_1_bn_2_beta  = o.modules[4].modules[0].modules[0].modules[0].modules[4].bias
    conv2_1_bn_2_mean  = o.modules[4].modules[0].modules[0].modules[0].modules[4].running_mean
    conv2_1_bn_2_var   = o.modules[4].modules[0].modules[0].modules[0].modules[4].running_var
    conv2_2_weights_1  = o.modules[4].modules[1].modules[0].modules[0].modules[0].weight
    conv2_2_bn_1_gamma = o.modules[4].modules[1].modules[0].modules[0].modules[1].weight
    conv2_2_bn_1_beta  = o.modules[4].modules[1].modules[0].modules[0].modules[1].bias
    conv2_2_bn_1_mean  = o.modules[4].modules[1].modules[0].modules[0].modules[1].running_mean
    conv2_2_bn_1_var   = o.modules[4].modules[1].modules[0].modules[0].modules[1].running_var
    conv2_2_weights_2  = o.modules[4].modules[1].modules[0].modules[0].modules[3].weight
    conv2_2_bn_2_gamma = o.modules[4].modules[1].modules[0].modules[0].modules[4].weight
    conv2_2_bn_2_beta  = o.modules[4].modules[1].modules[0].modules[0].modules[4].bias
    conv2_2_bn_2_mean  = o.modules[4].modules[1].modules[0].modules[0].modules[4].running_mean
    conv2_2_bn_2_var   = o.modules[4].modules[1].modules[0].modules[0].modules[4].running_var

    conv3_1_weights_skip = o.modules[5].modules[0].modules[0].modules[1].weight
    conv3_1_weights_1  = o.modules[5].modules[0].modules[0].modules[0].modules[0].weight
    conv3_1_bn_1_gamma = o.modules[5].modules[0].modules[0].modules[0].modules[1].weight
    conv3_1_bn_1_beta  = o.modules[5].modules[0].modules[0].modules[0].modules[1].bias
    conv3_1_bn_1_mean  = o.modules[5].modules[0].modules[0].modules[0].modules[1].running_mean
    conv3_1_bn_1_var   = o.modules[5].modules[0].modules[0].modules[0].modules[1].running_var
    conv3_1_weights_2  = o.modules[5].modules[0].modules[0].modules[0].modules[3].weight
    conv3_1_bn_2_gamma = o.modules[5].modules[0].modules[0].modules[0].modules[4].weight
    conv3_1_bn_2_beta  = o.modules[5].modules[0].modules[0].modules[0].modules[4].bias
    conv3_1_bn_2_mean  = o.modules[5].modules[0].modules[0].modules[0].modules[4].running_mean
    conv3_1_bn_2_var   = o.modules[5].modules[0].modules[0].modules[0].modules[4].running_var
    conv3_2_weights_1  = o.modules[5].modules[1].modules[0].modules[0].modules[0].weight
    conv3_2_bn_1_gamma = o.modules[5].modules[1].modules[0].modules[0].modules[1].weight
    conv3_2_bn_1_beta  = o.modules[5].modules[1].modules[0].modules[0].modules[1].bias
    conv3_2_bn_1_mean  = o.modules[5].modules[1].modules[0].modules[0].modules[1].running_mean
    conv3_2_bn_1_var   = o.modules[5].modules[1].modules[0].modules[0].modules[1].running_var
    conv3_2_weights_2  = o.modules[5].modules[1].modules[0].modules[0].modules[3].weight
    conv3_2_bn_2_gamma = o.modules[5].modules[1].modules[0].modules[0].modules[4].weight
    conv3_2_bn_2_beta  = o.modules[5].modules[1].modules[0].modules[0].modules[4].bias
    conv3_2_bn_2_mean  = o.modules[5].modules[1].modules[0].modules[0].modules[4].running_mean
    conv3_2_bn_2_var   = o.modules[5].modules[1].modules[0].modules[0].modules[4].running_var

    conv4_1_weights_skip = o.modules[6].modules[0].modules[0].modules[1].weight
    conv4_1_weights_1  = o.modules[6].modules[0].modules[0].modules[0].modules[0].weight
    conv4_1_bn_1_gamma = o.modules[6].modules[0].modules[0].modules[0].modules[1].weight
    conv4_1_bn_1_beta  = o.modules[6].modules[0].modules[0].modules[0].modules[1].bias
    conv4_1_bn_1_mean  = o.modules[6].modules[0].modules[0].modules[0].modules[1].running_mean
    conv4_1_bn_1_var   = o.modules[6].modules[0].modules[0].modules[0].modules[1].running_var
    conv4_1_weights_2  = o.modules[6].modules[0].modules[0].modules[0].modules[3].weight
    conv4_1_bn_2_gamma = o.modules[6].modules[0].modules[0].modules[0].modules[4].weight
    conv4_1_bn_2_beta  = o.modules[6].modules[0].modules[0].modules[0].modules[4].bias
    conv4_1_bn_2_mean  = o.modules[6].modules[0].modules[0].modules[0].modules[4].running_mean
    conv4_1_bn_2_var   = o.modules[6].modules[0].modules[0].modules[0].modules[4].running_var
    conv4_2_weights_1  = o.modules[6].modules[1].modules[0].modules[0].modules[0].weight
    conv4_2_bn_1_gamma = o.modules[6].modules[1].modules[0].modules[0].modules[1].weight
    conv4_2_bn_1_beta  = o.modules[6].modules[1].modules[0].modules[0].modules[1].bias
    conv4_2_bn_1_mean  = o.modules[6].modules[1].modules[0].modules[0].modules[1].running_mean
    conv4_2_bn_1_var   = o.modules[6].modules[1].modules[0].modules[0].modules[1].running_var
    conv4_2_weights_2  = o.modules[6].modules[1].modules[0].modules[0].modules[3].weight
    conv4_2_bn_2_gamma = o.modules[6].modules[1].modules[0].modules[0].modules[4].weight
    conv4_2_bn_2_beta  = o.modules[6].modules[1].modules[0].modules[0].modules[4].bias
    conv4_2_bn_2_mean  = o.modules[6].modules[1].modules[0].modules[0].modules[4].running_mean
    conv4_2_bn_2_var   = o.modules[6].modules[1].modules[0].modules[0].modules[4].running_var

    conv5_1_weights_skip = o.modules[7].modules[0].modules[0].modules[1].weight
    conv5_1_weights_1  = o.modules[7].modules[0].modules[0].modules[0].modules[0].weight
    conv5_1_bn_1_gamma = o.modules[7].modules[0].modules[0].modules[0].modules[1].weight
    conv5_1_bn_1_beta  = o.modules[7].modules[0].modules[0].modules[0].modules[1].bias
    conv5_1_bn_1_mean  = o.modules[7].modules[0].modules[0].modules[0].modules[1].running_mean
    conv5_1_bn_1_var   = o.modules[7].modules[0].modules[0].modules[0].modules[1].running_var
    conv5_1_weights_2  = o.modules[7].modules[0].modules[0].modules[0].modules[3].weight
    conv5_1_bn_2_gamma = o.modules[7].modules[0].modules[0].modules[0].modules[4].weight
    conv5_1_bn_2_beta  = o.modules[7].modules[0].modules[0].modules[0].modules[4].bias
    conv5_1_bn_2_mean  = o.modules[7].modules[0].modules[0].modules[0].modules[4].running_mean
    conv5_1_bn_2_var   = o.modules[7].modules[0].modules[0].modules[0].modules[4].running_var
    conv5_2_weights_1  = o.modules[7].modules[1].modules[0].modules[0].modules[0].weight
    conv5_2_bn_1_gamma = o.modules[7].modules[1].modules[0].modules[0].modules[1].weight
    conv5_2_bn_1_beta  = o.modules[7].modules[1].modules[0].modules[0].modules[1].bias
    conv5_2_bn_1_mean  = o.modules[7].modules[1].modules[0].modules[0].modules[1].running_mean
    conv5_2_bn_1_var   = o.modules[7].modules[1].modules[0].modules[0].modules[1].running_var
    conv5_2_weights_2  = o.modules[7].modules[1].modules[0].modules[0].modules[3].weight
    conv5_2_bn_2_gamma = o.modules[7].modules[1].modules[0].modules[0].modules[4].weight
    conv5_2_bn_2_beta  = o.modules[7].modules[1].modules[0].modules[0].modules[4].bias
    conv5_2_bn_2_mean  = o.modules[7].modules[1].modules[0].modules[0].modules[4].running_mean
    conv5_2_bn_2_var   = o.modules[7].modules[1].modules[0].modules[0].modules[4].running_var

    fc_weights = o.modules[10].weight
    fc_biases = o.modules[10].bias

    resnet18_weights = {
        'conv1/conv/kernel': conv1_weights,
        'conv1/bn/mu': conv1_bn_mean,
        'conv1/bn/sigma': conv1_bn_var,
        'conv1/bn/beta': conv1_bn_beta,
        'conv1/bn/gamma': conv1_bn_gamma,

        'conv2_1/conv_1/kernel': conv2_1_weights_1,
        'conv2_1/bn_1/mu':       conv2_1_bn_1_mean,
        'conv2_1/bn_1/sigma':    conv2_1_bn_1_var,
        'conv2_1/bn_1/beta':     conv2_1_bn_1_beta,
        'conv2_1/bn_1/gamma':    conv2_1_bn_1_gamma,
        'conv2_1/conv_2/kernel': conv2_1_weights_2,
        'conv2_1/bn_2/mu':       conv2_1_bn_2_mean,
        'conv2_1/bn_2/sigma':    conv2_1_bn_2_var,
        'conv2_1/bn_2/beta':     conv2_1_bn_2_beta,
        'conv2_1/bn_2/gamma':    conv2_1_bn_2_gamma,
        'conv2_2/conv_1/kernel': conv2_2_weights_1,
        'conv2_2/bn_1/mu':       conv2_2_bn_1_mean,
        'conv2_2/bn_1/sigma':    conv2_2_bn_1_var,
        'conv2_2/bn_1/beta':     conv2_2_bn_1_beta,
        'conv2_2/bn_1/gamma':    conv2_2_bn_1_gamma,
        'conv2_2/conv_2/kernel': conv2_2_weights_2,
        'conv2_2/bn_2/mu':       conv2_2_bn_2_mean,
        'conv2_2/bn_2/sigma':    conv2_2_bn_2_var,
        'conv2_2/bn_2/beta':     conv2_2_bn_2_beta,
        'conv2_2/bn_2/gamma':    conv2_2_bn_2_gamma,

        'conv3_1/shortcut/kernel':  conv3_1_weights_skip,
        'conv3_1/conv_1/kernel': conv3_1_weights_1,
        'conv3_1/bn_1/mu':       conv3_1_bn_1_mean,
        'conv3_1/bn_1/sigma':    conv3_1_bn_1_var,
        'conv3_1/bn_1/beta':     conv3_1_bn_1_beta,
        'conv3_1/bn_1/gamma':    conv3_1_bn_1_gamma,
        'conv3_1/conv_2/kernel': conv3_1_weights_2,
        'conv3_1/bn_2/mu':       conv3_1_bn_2_mean,
        'conv3_1/bn_2/sigma':    conv3_1_bn_2_var,
        'conv3_1/bn_2/beta':     conv3_1_bn_2_beta,
        'conv3_1/bn_2/gamma':    conv3_1_bn_2_gamma,
        'conv3_2/conv_1/kernel': conv3_2_weights_1,
        'conv3_2/bn_1/mu':       conv3_2_bn_1_mean,
        'conv3_2/bn_1/sigma':    conv3_2_bn_1_var,
        'conv3_2/bn_1/beta':     conv3_2_bn_1_beta,
        'conv3_2/bn_1/gamma':    conv3_2_bn_1_gamma,
        'conv3_2/conv_2/kernel': conv3_2_weights_2,
        'conv3_2/bn_2/mu':       conv3_2_bn_2_mean,
        'conv3_2/bn_2/sigma':    conv3_2_bn_2_var,
        'conv3_2/bn_2/beta':     conv3_2_bn_2_beta,
        'conv3_2/bn_2/gamma':    conv3_2_bn_2_gamma,

        'conv4_1/shortcut/kernel':  conv4_1_weights_skip,
        'conv4_1/conv_1/kernel': conv4_1_weights_1,
        'conv4_1/bn_1/mu':       conv4_1_bn_1_mean,
        'conv4_1/bn_1/sigma':    conv4_1_bn_1_var,
        'conv4_1/bn_1/beta':     conv4_1_bn_1_beta,
        'conv4_1/bn_1/gamma':    conv4_1_bn_1_gamma,
        'conv4_1/conv_2/kernel': conv4_1_weights_2,
        'conv4_1/bn_2/mu':       conv4_1_bn_2_mean,
        'conv4_1/bn_2/sigma':    conv4_1_bn_2_var,
        'conv4_1/bn_2/beta':     conv4_1_bn_2_beta,
        'conv4_1/bn_2/gamma':    conv4_1_bn_2_gamma,
        'conv4_2/conv_1/kernel': conv4_2_weights_1,
        'conv4_2/bn_1/mu':       conv4_2_bn_1_mean,
        'conv4_2/bn_1/sigma':    conv4_2_bn_1_var,
        'conv4_2/bn_1/beta':     conv4_2_bn_1_beta,
        'conv4_2/bn_1/gamma':    conv4_2_bn_1_gamma,
        'conv4_2/conv_2/kernel': conv4_2_weights_2,
        'conv4_2/bn_2/mu':       conv4_2_bn_2_mean,
        'conv4_2/bn_2/sigma':    conv4_2_bn_2_var,
        'conv4_2/bn_2/beta':     conv4_2_bn_2_beta,
        'conv4_2/bn_2/gamma':    conv4_2_bn_2_gamma,

        'conv5_1/shortcut/kernel':  conv5_1_weights_skip,
        'conv5_1/conv_1/kernel': conv5_1_weights_1,
        'conv5_1/bn_1/mu':       conv5_1_bn_1_mean,
        'conv5_1/bn_1/sigma':    conv5_1_bn_1_var,
        'conv5_1/bn_1/beta':     conv5_1_bn_1_beta,
        'conv5_1/bn_1/gamma':    conv5_1_bn_1_gamma,
        'conv5_1/conv_2/kernel': conv5_1_weights_2,
        'conv5_1/bn_2/mu':       conv5_1_bn_2_mean,
        'conv5_1/bn_2/sigma':    conv5_1_bn_2_var,
        'conv5_1/bn_2/beta':     conv5_1_bn_2_beta,
        'conv5_1/bn_2/gamma':    conv5_1_bn_2_gamma,
        'conv5_2/conv_1/kernel': conv5_2_weights_1,
        'conv5_2/bn_1/mu':       conv5_2_bn_1_mean,
        'conv5_2/bn_1/sigma':    conv5_2_bn_1_var,
        'conv5_2/bn_1/beta':     conv5_2_bn_1_beta,
        'conv5_2/bn_1/gamma':    conv5_2_bn_1_gamma,
        'conv5_2/conv_2/kernel': conv5_2_weights_2,
        'conv5_2/bn_2/mu':       conv5_2_bn_2_mean,
        'conv5_2/bn_2/sigma':    conv5_2_bn_2_var,
        'conv5_2/bn_2/beta':     conv5_2_bn_2_beta,
        'conv5_2/bn_2/gamma':    conv5_2_bn_2_gamma,

        #'logits/fc/weights': fc_weights,
        #'logits/fc/biases': fc_biases,
    }

    # Transpose conv and fc weights
    model_weights = {}
    for k, v in resnet18_weights.items():
        if len(v.shape) == 4:
            resnet18_weights[k] = np.transpose(v, (2, 3, 1, 0))
        elif len(v.shape) == 2:
            resnet18_weights[k] = np.transpose(v)
        else:
            resnet18_weights[k] = v

    # Save
    np.save('resnet18_weights.npy', resnet18_weights)

def import_resnet101_weights():

    resnet_101_weights = np.load("ResNet101.npy")
    
    resnet101_weights = {
    'model/conv1/conv/kernel': resnet_101_weights.item().get('conv1/W'),
    #'model/conv1/conv/bias':
    #'model/conv1/gn/beta':
    #'model/conv1/gn/gamma':
    'model/conv2_1/shortcut/kernel': resnet_101_weights.item().get('res2a_branch1/W'),
    #'model/conv2_1/shortcut/bias':
    'model/conv2_1/conv_1/kernel': resnet_101_weights.item().get('res2a_branch2a/W'),
    #'model/conv2_1/conv_1/bias':
    #'model/conv2_1/gn_1/beta':
    #'model/conv2_1/gn_1/gamma':
    'model/conv2_1/conv_2/kernel': resnet_101_weights.item().get('res2a_branch2b/W'),
    #'model/conv2_1/conv_2/bias':
    #'model/conv2_1/gn_2/beta':
    #'model/conv2_1/gn_2/gamma':
    'model/conv2_1/conv_3/kernel': resnet_101_weights.item().get('res2a_branch2c/W'),
    #'model/conv2_1/conv_3/bias':
    #'model/conv2_1/gn_3/beta':
    #'model/conv2_1/gn_3/gamma':
    'model/conv2_2/conv_1/kernel': resnet_101_weights.item().get('res2b_branch2a/W'),
    #'model/conv2_2/conv_1/bias':
    #'model/conv2_2/gn_1/beta':
    #'model/conv2_2/gn_1/gamma':
    'model/conv2_2/conv_2/kernel': resnet_101_weights.item().get('res2b_branch2b/W'),
    #'model/conv2_2/conv_2/bias':
    #'model/conv2_2/gn_2/beta':
    #'model/conv2_2/gn_2/gamma':
    'model/conv2_2/conv_3/kernel': resnet_101_weights.item().get('res2b_branch2c/W'),
    #'model/conv2_2/conv_3/bias':
    #'model/conv2_2/gn_3/beta':
    #'model/conv2_2/gn_3/gamma':
    'model/conv2_3/conv_1/kernel': resnet_101_weights.item().get('res2c_branch2a/W'),
    #'model/conv2_3/conv_1/bias':
    #'model/conv2_3/gn_1/beta':
    #'model/conv2_3/gn_1/gamma':
    'model/conv2_3/conv_2/kernel': resnet_101_weights.item().get('res2c_branch2b/W'),
    #'model/conv2_3/conv_2/bias':
    #'model/conv2_3/gn_2/beta':
    #'model/conv2_3/gn_2/gamma':
    'model/conv2_3/conv_3/kernel': resnet_101_weights.item().get('res2c_branch2c/W'),
    #'model/conv2_3/conv_3/bias':
    #'model/conv2_3/gn_3/beta':
    #'model/conv2_3/gn_3/gamma':
    'model/conv3_1/shortcut/kernel': resnet_101_weights.item().get('res3a_branch1/W'),
    #'model/conv3_1/shortcut/bias':
    'model/conv3_1/conv_1/kernel': resnet_101_weights.item().get('res3a_branch2a/W'),
    #'model/conv3_1/conv_1/bias':
    #'model/conv3_1/gn_1/beta':
    #'model/conv3_1/gn_1/gamma':
    'model/conv3_1/conv_2/kernel': resnet_101_weights.item().get('res3a_branch2b/W'),
    #'model/conv3_1/conv_2/bias':
    #'model/conv3_1/gn_2/beta':
    #'model/conv3_1/gn_2/gamma':
    'model/conv3_1/conv_3/kernel': resnet_101_weights.item().get('res3a_branch2c/W'),
    #'model/conv3_1/conv_3/bias':
    #'model/conv3_1/gn_3/beta':
    #'model/conv3_1/gn_3/gamma':
    'model/conv3_2/conv_1/kernel': resnet_101_weights.item().get('res3b1_branch2a/W'),
    #'model/conv3_2/conv_1/bias':
    #'model/conv3_2/gn_1/beta':
    #'model/conv3_2/gn_1/gamma':
    'model/conv3_2/conv_2/kernel': resnet_101_weights.item().get('res3b1_branch2b/W'),
    #'model/conv3_2/conv_2/bias':
    #'model/conv3_2/gn_2/beta':
    #'model/conv3_2/gn_2/gamma':
    'model/conv3_2/conv_3/kernel': resnet_101_weights.item().get('res3b1_branch2c/W'),
    #'model/conv3_2/conv_3/bias':
    #'model/conv3_2/gn_3/beta':
    #'model/conv3_2/gn_3/gamma':
    'model/conv3_3/conv_1/kernel': resnet_101_weights.item().get('res3b2_branch2a/W'),
    #'model/conv3_3/conv_1/bias':
    #'model/conv3_3/gn_1/beta':
    #'model/conv3_3/gn_1/gamma':
    'model/conv3_3/conv_2/kernel': resnet_101_weights.item().get('res3b2_branch2b/W'),
    #'model/conv3_3/conv_2/bias':
    #'model/conv3_3/gn_2/beta':
    #'model/conv3_3/gn_2/gamma':
    'model/conv3_3/conv_3/kernel': resnet_101_weights.item().get('res3b2_branch2c/W'),
    #'model/conv3_3/conv_3/bias':
    #'model/conv3_3/gn_3/beta':
    #'model/conv3_3/gn_3/gamma':
    'model/conv3_4/conv_1/kernel': resnet_101_weights.item().get('res3b3_branch2a/W'),
    #'model/conv3_4/conv_1/bias':
    #'model/conv3_4/gn_1/beta':
    #'model/conv3_4/gn_1/gamma':
    'model/conv3_4/conv_2/kernel': resnet_101_weights.item().get('res3b3_branch2b/W'),
    #'model/conv3_4/conv_2/bias':
    #'model/conv3_4/gn_2/beta':
    #'model/conv3_4/gn_2/gamma':
    'model/conv3_4/conv_3/kernel': resnet_101_weights.item().get('res3b3_branch2c/W'),
    #'model/conv3_4/conv_3/bias':
    #'model/conv3_4/gn_3/beta':
    #'model/conv3_4/gn_3/gamma':
    'model/conv4_1/shortcut/kernel': resnet_101_weights.item().get('res4a_branch1/W'),
    #'model/conv4_1/shortcut/bias':
    'model/conv4_1/conv_1/kernel': resnet_101_weights.item().get('res4a_branch2a/W'),
    #'model/conv4_1/conv_1/bias':
    #'model/conv4_1/gn_1/beta':
    #'model/conv4_1/gn_1/gamma':
    'model/conv4_1/conv_2/kernel': resnet_101_weights.item().get('res4a_branch2b/W'),
    #'model/conv4_1/conv_2/bias':
    #'model/conv4_1/gn_2/beta':
    #'model/conv4_1/gn_2/gamma':
    'model/conv4_1/conv_3/kernel': resnet_101_weights.item().get('res4a_branch2c/W'),
    #'model/conv4_1/conv_3/bias':
    #'model/conv4_1/gn_3/beta':
    #'model/conv4_1/gn_3/gamma':
    'model/conv4_2/conv_1/kernel': resnet_101_weights.item().get('res4b1_branch2a/W'),
    #'model/conv4_2/conv_1/bias':
    #'model/conv4_2/gn_1/beta':
    #'model/conv4_2/gn_1/gamma':
    'model/conv4_2/conv_2/kernel': resnet_101_weights.item().get('res4b1_branch2b/W'),
    #'model/conv4_2/conv_2/bias':
    #'model/conv4_2/gn_2/beta':
    #'model/conv4_2/gn_2/gamma':
    'model/conv4_2/conv_3/kernel': resnet_101_weights.item().get('res4b1_branch2c/W'),
    #'model/conv4_2/conv_3/bias':
    #'model/conv4_2/gn_3/beta':
    #'model/conv4_2/gn_3/gamma':
    'model/conv4_3/conv_1/kernel': resnet_101_weights.item().get('res4b2_branch2a/W'),
    #'model/conv4_3/conv_1/bias':
    #'model/conv4_3/gn_1/beta':
    #'model/conv4_3/gn_1/gamma':
    'model/conv4_3/conv_2/kernel': resnet_101_weights.item().get('res4b2_branch2b/W'),
    #'model/conv4_3/conv_2/bias':
    #'model/conv4_3/gn_2/beta':
    #'model/conv4_3/gn_2/gamma':
    'model/conv4_3/conv_3/kernel': resnet_101_weights.item().get('res4b2_branch2c/W'),
    #'model/conv4_3/conv_3/bias':
    #'model/conv4_3/gn_3/beta':
    #'model/conv4_3/gn_3/gamma':
    'model/conv4_4/conv_1/kernel': resnet_101_weights.item().get('res4b3_branch2a/W'),
    #'model/conv4_4/conv_1/bias':
    #'model/conv4_4/gn_1/beta':
    #'model/conv4_4/gn_1/gamma':
    'model/conv4_4/conv_2/kernel': resnet_101_weights.item().get('res4b3_branch2b/W'),
    #'model/conv4_4/conv_2/bias':
    #'model/conv4_4/gn_2/beta':
    #'model/conv4_4/gn_2/gamma':
    'model/conv4_4/conv_3/kernel': resnet_101_weights.item().get('res4b3_branch2c/W'),
    #'model/conv4_4/conv_3/bias':
    #'model/conv4_4/gn_3/beta':
    #'model/conv4_4/gn_3/gamma':
    'model/conv4_5/conv_1/kernel': resnet_101_weights.item().get('res4b4_branch2a/W'),
    #'model/conv4_5/conv_1/bias':
    #'model/conv4_5/gn_1/beta':
    #'model/conv4_5/gn_1/gamma':
    'model/conv4_5/conv_2/kernel': resnet_101_weights.item().get('res4b4_branch2b/W'),
    #'model/conv4_5/conv_2/bias':
    #'model/conv4_5/gn_2/beta':
    #'model/conv4_5/gn_2/gamma':
    'model/conv4_5/conv_3/kernel': resnet_101_weights.item().get('res4b4_branch2c/W'),
    #'model/conv4_5/conv_3/bias':
    #'model/conv4_5/gn_3/beta':
    #'model/conv4_5/gn_3/gamma':
    'model/conv4_6/conv_1/kernel': resnet_101_weights.item().get('res4b5_branch2a/W'),
    #'model/conv4_6/conv_1/bias':
    #'model/conv4_6/gn_1/beta':
    #'model/conv4_6/gn_1/gamma':
    'model/conv4_6/conv_2/kernel': resnet_101_weights.item().get('res4b5_branch2b/W'),
    #'model/conv4_6/conv_2/bias':
    #'model/conv4_6/gn_2/beta':
    #'model/conv4_6/gn_2/gamma':
    'model/conv4_6/conv_3/kernel': resnet_101_weights.item().get('res4b5_branch2c/W'),
    #'model/conv4_6/conv_3/bias':
    #'model/conv4_6/gn_3/beta':
    #'model/conv4_6/gn_3/gamma':
    'model/conv4_7/conv_1/kernel': resnet_101_weights.item().get('res4b6_branch2a/W'),
    #'model/conv4_7/conv_1/bias':
    #'model/conv4_7/gn_1/beta':
    #'model/conv4_7/gn_1/gamma':
    'model/conv4_7/conv_2/kernel': resnet_101_weights.item().get('res4b6_branch2b/W'),
    #'model/conv4_7/conv_2/bias':
    #'model/conv4_7/gn_2/beta':
    #'model/conv4_7/gn_2/gamma':
    'model/conv4_7/conv_3/kernel': resnet_101_weights.item().get('res4b6_branch2c/W'),
    #'model/conv4_7/conv_3/bias':
    #'model/conv4_7/gn_3/beta':
    #'model/conv4_7/gn_3/gamma':
    'model/conv4_8/conv_1/kernel': resnet_101_weights.item().get('res4b7_branch2a/W'),
    #'model/conv4_8/conv_1/bias':
    #'model/conv4_8/gn_1/beta':
    #'model/conv4_8/gn_1/gamma':
    'model/conv4_8/conv_2/kernel': resnet_101_weights.item().get('res4b7_branch2b/W'),
    #'model/conv4_8/conv_2/bias':
    #'model/conv4_8/gn_2/beta':
    #'model/conv4_8/gn_2/gamma':
    'model/conv4_8/conv_3/kernel': resnet_101_weights.item().get('res4b7_branch2c/W'),
    #'model/conv4_8/conv_3/bias':
    #'model/conv4_8/gn_3/beta':
    #'model/conv4_8/gn_3/gamma':
    'model/conv4_9/conv_1/kernel': resnet_101_weights.item().get('res4b8_branch2a/W'),
    #'model/conv4_9/conv_1/bias':
    #'model/conv4_9/gn_1/beta':
    #'model/conv4_9/gn_1/gamma':
    'model/conv4_9/conv_2/kernel': resnet_101_weights.item().get('res4b8_branch2b/W'),
    #'model/conv4_9/conv_2/bias':
    #'model/conv4_9/gn_2/beta':
    #'model/conv4_9/gn_2/gamma':
    'model/conv4_9/conv_3/kernel': resnet_101_weights.item().get('res4b8_branch2c/W'),
    #'model/conv4_9/conv_3/bias':
    #'model/conv4_9/gn_3/beta':
    #'model/conv4_9/gn_3/gamma':
    'model/conv4_10/conv_1/kernel': resnet_101_weights.item().get('res4b9_branch2a/W'),
    #'model/conv4_10/conv_1/bias':
    #'model/conv4_10/gn_1/beta':
    #'model/conv4_10/gn_1/gamma':
    'model/conv4_10/conv_2/kernel': resnet_101_weights.item().get('res4b9_branch2b/W'),
    #'model/conv4_10/conv_2/bias':
    #'model/conv4_10/gn_2/beta':
    #'model/conv4_10/gn_2/gamma':
    'model/conv4_10/conv_3/kernel': resnet_101_weights.item().get('res4b9_branch2c/W'),
    #'model/conv4_10/conv_3/bias':
    #'model/conv4_10/gn_3/beta':
    #'model/conv4_10/gn_3/gamma':
    'model/conv4_11/conv_1/kernel': resnet_101_weights.item().get('res4b10_branch2a/W'),
    #'model/conv4_11/conv_1/bias':
    #'model/conv4_11/gn_1/beta':
    #'model/conv4_11/gn_1/gamma':
    'model/conv4_11/conv_2/kernel': resnet_101_weights.item().get('res4b10_branch2b/W'),
    #'model/conv4_11/conv_2/bias':
    #'model/conv4_11/gn_2/beta':
    #'model/conv4_11/gn_2/gamma':
    'model/conv4_11/conv_3/kernel': resnet_101_weights.item().get('res4b10_branch2c/W'),
    #'model/conv4_11/conv_3/bias':
    #'model/conv4_11/gn_3/beta':
    #'model/conv4_11/gn_3/gamma':
    'model/conv4_12/conv_1/kernel': resnet_101_weights.item().get('res4b11_branch2a/W'),
    #'model/conv4_12/conv_1/bias':
    #'model/conv4_12/gn_1/beta':
    #'model/conv4_12/gn_1/gamma':
    'model/conv4_12/conv_2/kernel': resnet_101_weights.item().get('res4b11_branch2b/W'),
    #'model/conv4_12/conv_2/bias':
    #'model/conv4_12/gn_2/beta':
    #'model/conv4_12/gn_2/gamma':
    'model/conv4_12/conv_3/kernel': resnet_101_weights.item().get('res4b11_branch2c/W'),
    #'model/conv4_12/conv_3/bias':
    #'model/conv4_12/gn_3/beta':
    #'model/conv4_12/gn_3/gamma':
    'model/conv4_13/conv_1/kernel': resnet_101_weights.item().get('res4b12_branch2a/W'),
    #'model/conv4_13/conv_1/bias':
    #'model/conv4_13/gn_1/beta':
    #'model/conv4_13/gn_1/gamma':
    'model/conv4_13/conv_2/kernel': resnet_101_weights.item().get('res4b12_branch2b/W'),
    #'model/conv4_13/conv_2/bias':
    #'model/conv4_13/gn_2/beta':
    #'model/conv4_13/gn_2/gamma':
    'model/conv4_13/conv_3/kernel': resnet_101_weights.item().get('res4b12_branch2c/W'),
    #'model/conv4_13/conv_3/bias':
    #'model/conv4_13/gn_3/beta':
    #'model/conv4_13/gn_3/gamma':
    'model/conv4_14/conv_1/kernel': resnet_101_weights.item().get('res4b13_branch2a/W'),
    #'model/conv4_14/conv_1/bias':
    #'model/conv4_14/gn_1/beta':
    #'model/conv4_14/gn_1/gamma':
    'model/conv4_14/conv_2/kernel': resnet_101_weights.item().get('res4b13_branch2b/W'),
    #'model/conv4_14/conv_2/bias':
    #'model/conv4_14/gn_2/beta':
    #'model/conv4_14/gn_2/gamma':
    'model/conv4_14/conv_3/kernel': resnet_101_weights.item().get('res4b13_branch2c/W'),
    #'model/conv4_14/conv_3/bias':
    #'model/conv4_14/gn_3/beta':
    #'model/conv4_14/gn_3/gamma':
    'model/conv4_15/conv_1/kernel': resnet_101_weights.item().get('res4b14_branch2a/W'),
    #'model/conv4_15/conv_1/bias':
    #'model/conv4_15/gn_1/beta':
    #'model/conv4_15/gn_1/gamma':
    'model/conv4_15/conv_2/kernel': resnet_101_weights.item().get('res4b14_branch2b/W'),
    #'model/conv4_15/conv_2/bias':
    #'model/conv4_15/gn_2/beta':
    #'model/conv4_15/gn_2/gamma':
    'model/conv4_15/conv_3/kernel': resnet_101_weights.item().get('res4b14_branch2c/W'),
    #'model/conv4_15/conv_3/bias':
    #'model/conv4_15/gn_3/beta':
    #'model/conv4_15/gn_3/gamma':
    'model/conv4_16/conv_1/kernel': resnet_101_weights.item().get('res4b15_branch2a/W'),
    #'model/conv4_16/conv_1/bias':
    #'model/conv4_16/gn_1/beta':
    #'model/conv4_16/gn_1/gamma':
    'model/conv4_16/conv_2/kernel': resnet_101_weights.item().get('res4b15_branch2b/W'),
    #'model/conv4_16/conv_2/bias':
    #'model/conv4_16/gn_2/beta':
    #'model/conv4_16/gn_2/gamma':
    'model/conv4_16/conv_3/kernel': resnet_101_weights.item().get('res4b15_branch2c/W'),
    #'model/conv4_16/conv_3/bias':
    #'model/conv4_16/gn_3/beta':
    #'model/conv4_16/gn_3/gamma':
    'model/conv4_17/conv_1/kernel': resnet_101_weights.item().get('res4b16_branch2a/W'),
    #'model/conv4_17/conv_1/bias':
    #'model/conv4_17/gn_1/beta':
    #'model/conv4_17/gn_1/gamma':
    'model/conv4_17/conv_2/kernel': resnet_101_weights.item().get('res4b16_branch2b/W'),
    #'model/conv4_17/conv_2/bias':
    #'model/conv4_17/gn_2/beta':
    #'model/conv4_17/gn_2/gamma':
    'model/conv4_17/conv_3/kernel': resnet_101_weights.item().get('res4b16_branch2c/W'),
    #'model/conv4_17/conv_3/bias':
    #'model/conv4_17/gn_3/beta':
    #'model/conv4_17/gn_3/gamma':
    'model/conv4_18/conv_1/kernel': resnet_101_weights.item().get('res4b17_branch2a/W'),
    #'model/conv4_18/conv_1/bias':
    #'model/conv4_18/gn_1/beta':
    #'model/conv4_18/gn_1/gamma':
    'model/conv4_18/conv_2/kernel': resnet_101_weights.item().get('res4b17_branch2b/W'),
    #'model/conv4_18/conv_2/bias':
    #'model/conv4_18/gn_2/beta':
    #'model/conv4_18/gn_2/gamma':
    'model/conv4_18/conv_3/kernel': resnet_101_weights.item().get('res4b17_branch2c/W'),
    #'model/conv4_18/conv_3/bias':
    #'model/conv4_18/gn_3/beta':
    #'model/conv4_18/gn_3/gamma':
    'model/conv4_19/conv_1/kernel': resnet_101_weights.item().get('res4b18_branch2a/W'),
    #'model/conv4_19/conv_1/bias':
    #'model/conv4_19/gn_1/beta':
    #'model/conv4_19/gn_1/gamma':
    'model/conv4_19/conv_2/kernel': resnet_101_weights.item().get('res4b18_branch2b/W'),
    #'model/conv4_19/conv_2/bias':
    #'model/conv4_19/gn_2/beta':
    #'model/conv4_19/gn_2/gamma':
    'model/conv4_19/conv_3/kernel': resnet_101_weights.item().get('res4b18_branch2c/W'),
    #'model/conv4_19/conv_3/bias':
    #'model/conv4_19/gn_3/beta':
    #'model/conv4_19/gn_3/gamma':
    'model/conv4_20/conv_1/kernel': resnet_101_weights.item().get('res4b19_branch2a/W'),
    #'model/conv4_20/conv_1/bias':
    #'model/conv4_20/gn_1/beta':
    #'model/conv4_20/gn_1/gamma':
    'model/conv4_20/conv_2/kernel': resnet_101_weights.item().get('res4b19_branch2b/W'),
    #'model/conv4_20/conv_2/bias':
    #'model/conv4_20/gn_2/beta':
    #'model/conv4_20/gn_2/gamma':
    'model/conv4_20/conv_3/kernel': resnet_101_weights.item().get('res4b19_branch2c/W'),
    #'model/conv4_20/conv_3/bias':
    #'model/conv4_20/gn_3/beta':
    #'model/conv4_20/gn_3/gamma':
    'model/conv4_21/conv_1/kernel': resnet_101_weights.item().get('res4b20_branch2a/W'),
    #'model/conv4_21/conv_1/bias':
    #'model/conv4_21/gn_1/beta':
    #'model/conv4_21/gn_1/gamma':
    'model/conv4_21/conv_2/kernel': resnet_101_weights.item().get('res4b20_branch2b/W'),
    #'model/conv4_21/conv_2/bias':
    #'model/conv4_21/gn_2/beta':
    #'model/conv4_21/gn_2/gamma':
    'model/conv4_21/conv_3/kernel': resnet_101_weights.item().get('res4b20_branch2c/W'),
    #'model/conv4_21/conv_3/bias':
    #'model/conv4_21/gn_3/beta':
    #'model/conv4_21/gn_3/gamma':
    'model/conv4_22/conv_1/kernel': resnet_101_weights.item().get('res4b21_branch2a/W'),
    #'model/conv4_22/conv_1/bias':
    #'model/conv4_22/gn_1/beta':
    #'model/conv4_22/gn_1/gamma':
    'model/conv4_22/conv_2/kernel': resnet_101_weights.item().get('res4b21_branch2b/W'),
    #'model/conv4_22/conv_2/bias':
    #'model/conv4_22/gn_2/beta':
    #'model/conv4_22/gn_2/gamma':
    'model/conv4_22/conv_3/kernel': resnet_101_weights.item().get('res4b21_branch2c/W'),
    #'model/conv4_22/conv_3/bias':
    #'model/conv4_22/gn_3/beta':
    #'model/conv4_22/gn_3/gamma':
    'model/conv4_23/conv_1/kernel': resnet_101_weights.item().get('res4b22_branch2a/W'),
    #'model/conv4_23/conv_1/bias':
    #'model/conv4_23/gn_1/beta':
    #'model/conv4_23/gn_1/gamma':
    'model/conv4_23/conv_2/kernel': resnet_101_weights.item().get('res4b22_branch2b/W'),
    #'model/conv4_23/conv_2/bias':
    #'model/conv4_23/gn_2/beta':
    #'model/conv4_23/gn_2/gamma':
    'model/conv4_23/conv_3/kernel': resnet_101_weights.item().get('res4b22_branch2c/W'),
    #'model/conv4_23/conv_3/bias':
    #'model/conv4_23/gn_3/beta':
    #'model/conv4_23/gn_3/gamma':
    'model/conv5_1/shortcut/kernel': resnet_101_weights.item().get('res5a_branch1/W'),
    #'model/conv5_1/shortcut/bias':
    'model/conv5_1/conv_1/kernel': resnet_101_weights.item().get('res5a_branch2a/W'),
    #'model/conv5_1/conv_1/bias':
    #'model/conv5_1/gn_1/beta':
    #'model/conv5_1/gn_1/gamma':
    'model/conv5_1/conv_2/kernel': resnet_101_weights.item().get('res5a_branch2b/W'),
    #'model/conv5_1/conv_2/bias':
    #'model/conv5_1/gn_2/beta':
    #'model/conv5_1/gn_2/gamma':
    'model/conv5_1/conv_3/kernel': resnet_101_weights.item().get('res5a_branch2c/W'),
    #'model/conv5_1/conv_3/bias':
    #'model/conv5_1/gn_3/beta':
    #'model/conv5_1/gn_3/gamma':
    'model/conv5_2/conv_1/kernel': resnet_101_weights.item().get('res5b_branch2a/W'),
    #'model/conv5_2/conv_1/bias':
    #'model/conv5_2/gn_1/beta':
    #'model/conv5_2/gn_1/gamma':
    'model/conv5_2/conv_2/kernel': resnet_101_weights.item().get('res5b_branch2b/W'),
    #'model/conv5_2/conv_2/bias':
    #'model/conv5_2/gn_2/beta':
    #'model/conv5_2/gn_2/gamma':
    'model/conv5_2/conv_3/kernel': resnet_101_weights.item().get('res5b_branch2c/W'),
    #'model/conv5_2/conv_3/bias':
    #'model/conv5_2/gn_3/beta':
    #'model/conv5_2/gn_3/gamma':
    'model/conv5_3/conv_1/kernel': resnet_101_weights.item().get('res5c_branch2a/W'),
    #'model/conv5_3/conv_1/bias':
    #'model/conv5_3/gn_1/beta':
    #'model/conv5_3/gn_1/gamma':
    'model/conv5_3/conv_2/kernel': resnet_101_weights.item().get('res5c_branch2b/W'),
    #'model/conv5_3/conv_2/bias':
    #'model/conv5_3/gn_2/beta':
    #'model/conv5_3/gn_2/gamma':
    'model/conv5_3/conv_3/kernel': resnet_101_weights.item().get('res5c_branch2c/W'),
    #'model/conv5_3/conv_3/bias':
    #'model/conv5_3/gn_3/beta':
    #'model/conv5_3/gn_3/gamma':
    }

    # Transpose conv and fc weights. ???
    model_weights = {}
    #for k, v in resnet101_weights.items():
    #    if len(v.shape) == 4:
    #        resnet101_weights[k] = np.transpose(v, (2, 3, 1, 0))
    #    elif len(v.shape) == 2:
    #        resnet101_weights[k] = np.transpose(v)
    #    else:
    #        resnet18_weights[k] = v

    # Save
    np.save('resnet101_weights.npy', resnet101_weights)

if __name__ == "__main__":

    import_resnet101_weights()