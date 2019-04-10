#! /usr/bin/python
# -*- coding: utf8 -*-

import time
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

# from tensorflow.python.ops import variable_scope as vs
# from tensorflow.python.ops import math_ops, init_ops, array_ops, nn
# from tensorflow.python.util import nest
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell
# https://github.com/david-gpu/srez/blob/master/srez_model.py


def SRGAN_g(t_image, is_train=False, reuse=False):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    # random_normal_initializer：返回一个生成具有正态分布的张量的初始化器（参数：平均数、标准差）
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        # t_image.shape = [batch_size=16, 96, 96, 3]
        
        # 神经网络的基础是一个 InputLayer 实例。n代表了将要提供给网络的输入数据。
        n = InputLayer(t_image, name='in')
        # 卷积层+激励层rulu
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        temp = n
        
        # B residual blocks
        # 残差网络模块
        for i in range(16):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='b_residual_add/%s' % i)
            n = nn

        # 卷积层
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        
        # BN层 https://www.cnblogs.com/guoyaohua/p/8724433.html
        # 而BN就是通过一定的规范化手段，把每层的输入的分布强行拉回到均值为0方差为1的标准正态分布，加速收敛，加快训练
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        
        # Elementwise层，实现残差网络，将具有相同结构的网络层的神经元相加
        n = ElementwiseLayer([n, temp], tf.add, name='add3')
        # 残差网络模块结束
        # B residual blacks end

        # 卷积+上采样    SubpixelConv2d上采样：W * H * C*r*r -> W*r * H*r * C
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')

        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        return n
'''
[TL]   layer   0: t_image_input_to_SRGAN_generator:0 (16, 96, 96, 3)    float32
[TL]   layer   1: n64s1/c/Relu:0       (16, 96, 96, 64)    float32
[TL]   layer   2: n64s1/c1/0/Conv2D:0  (16, 96, 96, 64)    float32
[TL]   layer   3: n64s1/b1/0/Relu:0    (16, 96, 96, 64)    float32
[TL]   layer   4: n64s1/c2/0/Conv2D:0  (16, 96, 96, 64)    float32
[TL]   layer   5: n64s1/b2/0/batchnorm/Add_1:0 (16, 96, 96, 64)    float32
[TL]   layer   6: b_residual_add/0:0   (16, 96, 96, 64)    float32
[TL]   layer   7: n64s1/c1/1/Conv2D:0  (16, 96, 96, 64)    float32
[TL]   layer   8: n64s1/b1/1/Relu:0    (16, 96, 96, 64)    float32
[TL]   layer   9: n64s1/c2/1/Conv2D:0  (16, 96, 96, 64)    float32
[TL]   layer  10: n64s1/b2/1/batchnorm/Add_1:0 (16, 96, 96, 64)    float32
[TL]   layer  11: b_residual_add/1:0   (16, 96, 96, 64)    float32
[TL]   layer  12: n64s1/c1/2/Conv2D:0  (16, 96, 96, 64)    float32
[TL]   layer  13: n64s1/b1/2/Relu:0    (16, 96, 96, 64)    float32
[TL]   layer  14: n64s1/c2/2/Conv2D:0  (16, 96, 96, 64)    float32
[TL]   layer  15: n64s1/b2/2/batchnorm/Add_1:0 (16, 96, 96, 64)    float32
[TL]   layer  16: b_residual_add/2:0   (16, 96, 96, 64)    float32
[TL]   layer  17: n64s1/c1/3/Conv2D:0  (16, 96, 96, 64)    float32
[TL]   layer  18: n64s1/b1/3/Relu:0    (16, 96, 96, 64)    float32
[TL]   layer  19: n64s1/c2/3/Conv2D:0  (16, 96, 96, 64)    float32
[TL]   layer  20: n64s1/b2/3/batchnorm/Add_1:0 (16, 96, 96, 64)    float32
[TL]   layer  21: b_residual_add/3:0   (16, 96, 96, 64)    float32
[TL]   layer  22: n64s1/c1/4/Conv2D:0  (16, 96, 96, 64)    float32
[TL]   layer  23: n64s1/b1/4/Relu:0    (16, 96, 96, 64)    float32
[TL]   layer  24: n64s1/c2/4/Conv2D:0  (16, 96, 96, 64)    float32
[TL]   layer  25: n64s1/b2/4/batchnorm/Add_1:0 (16, 96, 96, 64)    float32
[TL]   layer  26: b_residual_add/4:0   (16, 96, 96, 64)    float32
[TL]   layer  27: n64s1/c1/5/Conv2D:0  (16, 96, 96, 64)    float32
[TL]   layer  28: n64s1/b1/5/Relu:0    (16, 96, 96, 64)    float32
[TL]   layer  29: n64s1/c2/5/Conv2D:0  (16, 96, 96, 64)    float32
[TL]   layer  30: n64s1/b2/5/batchnorm/Add_1:0 (16, 96, 96, 64)    float32
[TL]   layer  31: b_residual_add/5:0   (16, 96, 96, 64)    float32
[TL]   layer  32: n64s1/c1/6/Conv2D:0  (16, 96, 96, 64)    float32
[TL]   layer  33: n64s1/b1/6/Relu:0    (16, 96, 96, 64)    float32
[TL]   layer  34: n64s1/c2/6/Conv2D:0  (16, 96, 96, 64)    float32
[TL]   layer  35: n64s1/b2/6/batchnorm/Add_1:0 (16, 96, 96, 64)    float32
[TL]   layer  36: b_residual_add/6:0   (16, 96, 96, 64)    float32
[TL]   layer  37: n64s1/c1/7/Conv2D:0  (16, 96, 96, 64)    float32
[TL]   layer  38: n64s1/b1/7/Relu:0    (16, 96, 96, 64)    float32
[TL]   layer  39: n64s1/c2/7/Conv2D:0  (16, 96, 96, 64)    float32
[TL]   layer  40: n64s1/b2/7/batchnorm/Add_1:0 (16, 96, 96, 64)    float32
[TL]   layer  41: b_residual_add/7:0   (16, 96, 96, 64)    float32
[TL]   layer  42: n64s1/c1/8/Conv2D:0  (16, 96, 96, 64)    float32
[TL]   layer  43: n64s1/b1/8/Relu:0    (16, 96, 96, 64)    float32
[TL]   layer  44: n64s1/c2/8/Conv2D:0  (16, 96, 96, 64)    float32
[TL]   layer  45: n64s1/b2/8/batchnorm/Add_1:0 (16, 96, 96, 64)    float32
[TL]   layer  46: b_residual_add/8:0   (16, 96, 96, 64)    float32
[TL]   layer  47: n64s1/c1/9/Conv2D:0  (16, 96, 96, 64)    float32
[TL]   layer  48: n64s1/b1/9/Relu:0    (16, 96, 96, 64)    float32
[TL]   layer  49: n64s1/c2/9/Conv2D:0  (16, 96, 96, 64)    float32
[TL]   layer  50: n64s1/b2/9/batchnorm/Add_1:0 (16, 96, 96, 64)    float32
[TL]   layer  51: b_residual_add/9:0   (16, 96, 96, 64)    float32
[TL]   layer  52: n64s1/c1/10/Conv2D:0 (16, 96, 96, 64)    float32
[TL]   layer  53: n64s1/b1/10/Relu:0   (16, 96, 96, 64)    float32
[TL]   layer  54: n64s1/c2/10/Conv2D:0 (16, 96, 96, 64)    float32
[TL]   layer  55: n64s1/b2/10/batchnorm/Add_1:0 (16, 96, 96, 64)    float32
[TL]   layer  56: b_residual_add/10:0  (16, 96, 96, 64)    float32
[TL]   layer  57: n64s1/c1/11/Conv2D:0 (16, 96, 96, 64)    float32
[TL]   layer  58: n64s1/b1/11/Relu:0   (16, 96, 96, 64)    float32
[TL]   layer  59: n64s1/c2/11/Conv2D:0 (16, 96, 96, 64)    float32
[TL]   layer  60: n64s1/b2/11/batchnorm/Add_1:0 (16, 96, 96, 64)    float32
[TL]   layer  61: b_residual_add/11:0  (16, 96, 96, 64)    float32
[TL]   layer  62: n64s1/c1/12/Conv2D:0 (16, 96, 96, 64)    float32
[TL]   layer  63: n64s1/b1/12/Relu:0   (16, 96, 96, 64)    float32
[TL]   layer  64: n64s1/c2/12/Conv2D:0 (16, 96, 96, 64)    float32
[TL]   layer  65: n64s1/b2/12/batchnorm/Add_1:0 (16, 96, 96, 64)    float32
[TL]   layer  66: b_residual_add/12:0  (16, 96, 96, 64)    float32
[TL]   layer  67: n64s1/c1/13/Conv2D:0 (16, 96, 96, 64)    float32
[TL]   layer  68: n64s1/b1/13/Relu:0   (16, 96, 96, 64)    float32
[TL]   layer  69: n64s1/c2/13/Conv2D:0 (16, 96, 96, 64)    float32
[TL]   layer  70: n64s1/b2/13/batchnorm/Add_1:0 (16, 96, 96, 64)    float32
[TL]   layer  71: b_residual_add/13:0  (16, 96, 96, 64)    float32
[TL]   layer  72: n64s1/c1/14/Conv2D:0 (16, 96, 96, 64)    float32
[TL]   layer  73: n64s1/b1/14/Relu:0   (16, 96, 96, 64)    float32
[TL]   layer  74: n64s1/c2/14/Conv2D:0 (16, 96, 96, 64)    float32
[TL]   layer  75: n64s1/b2/14/batchnorm/Add_1:0 (16, 96, 96, 64)    float32
[TL]   layer  76: b_residual_add/14:0  (16, 96, 96, 64)    float32
[TL]   layer  77: n64s1/c1/15/Conv2D:0 (16, 96, 96, 64)    float32
[TL]   layer  78: n64s1/b1/15/Relu:0   (16, 96, 96, 64)    float32
[TL]   layer  79: n64s1/c2/15/Conv2D:0 (16, 96, 96, 64)    float32
[TL]   layer  80: n64s1/b2/15/batchnorm/Add_1:0 (16, 96, 96, 64)    float32
[TL]   layer  81: b_residual_add/15:0  (16, 96, 96, 64)    float32
[TL]   layer  82: n64s1/c/m/Conv2D:0   (16, 96, 96, 64)    float32
[TL]   layer  83: n64s1/b/m/batchnorm/Add_1:0 (16, 96, 96, 64)    float32
[TL]   layer  84: add3:0               (16, 96, 96, 64)    float32
[TL]   layer  85: n256s1/1/BiasAdd:0   (16, 96, 96, 256)    float32
[TL]   layer  86: pixelshufflerx2/1/Relu:0 (16, 192, 192, 64)    float32
[TL]   layer  87: n256s1/2/BiasAdd:0   (16, 192, 192, 256)    float32
[TL]   layer  88: pixelshufflerx2/2/Relu:0 (16, 384, 384, 64)    float32
[TL]   layer  89: out/Tanh:0           (16, 384, 384, 3)    float32
'''

def SRGAN_d(input_images, is_train=True, reuse=False):
    
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    
    df_dim = 64
    # Leaky Relu：x>0时y=x；x<0时y=x*a (a<1)
    # lambda作为一个表达式，定义了一个匿名函数，lrelu(x)...
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    
    with tf.variable_scope("SRGAN_d", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='input/images')
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init, name='h0/c')

        net_h1 = Conv2d(net_h0, df_dim * 2, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h1/c')
        net_h1 = BatchNormLayer(net_h1, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h1/bn')
        net_h2 = Conv2d(net_h1, df_dim * 4, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h2/c')
        net_h2 = BatchNormLayer(net_h2, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h2/bn')
        net_h3 = Conv2d(net_h2, df_dim * 8, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h3/c')
        net_h3 = BatchNormLayer(net_h3, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h3/bn')
        net_h4 = Conv2d(net_h3, df_dim * 16, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h4/c')
        net_h4 = BatchNormLayer(net_h4, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h4/bn')
        net_h5 = Conv2d(net_h4, df_dim * 32, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h5/c')
        net_h5 = BatchNormLayer(net_h5, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h5/bn')
        net_h6 = Conv2d(net_h5, df_dim * 16, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h6/c')
        net_h6 = BatchNormLayer(net_h6, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h6/bn')
        net_h7 = Conv2d(net_h6, df_dim * 8, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h7/c')
        net_h7 = BatchNormLayer(net_h7, is_train=is_train, gamma_init=gamma_init, name='h7/bn')

        net = Conv2d(net_h7, df_dim * 2, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res/c')
        net = BatchNormLayer(net, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='res/bn')
        net = Conv2d(net, df_dim * 2, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res/c2')
        net = BatchNormLayer(net, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='res/bn2')
        net = Conv2d(net, df_dim * 8, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res/c3')
        net = BatchNormLayer(net, is_train=is_train, gamma_init=gamma_init, name='res/bn3')
        net_h8 = ElementwiseLayer([net_h7, net], combine_fn=tf.add, name='res/add')
        net_h8.outputs = tl.act.lrelu(net_h8.outputs, 0.2)

        # Flatten层，reshape high-dimension input to a vector
        net_ho = FlattenLayer(net_h8, name='ho/flatten')
        # 全连接层；n_units=1输出神经元为1个 ；act=tf.identity输出等于输入
        net_ho = DenseLayer(net_ho, n_units=1, act=tf.identity, W_init=w_init, name='ho/dense')
        logits = net_ho.outputs
        # 激励函数为sigmoid
        net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)

    return net_ho, logits
'''
[TL]   layer   0: t_target_image:0     (16, 384, 384, 3)    float32
[TL]   layer   1: h0/c/leaky_relu:0    (16, 192, 192, 64)    float32
[TL]   layer   2: h1/c/Conv2D:0        (16, 96, 96, 128)    float32
[TL]   layer   3: h1/bn/leaky_relu:0   (16, 96, 96, 128)    float32
[TL]   layer   4: h2/c/Conv2D:0        (16, 48, 48, 256)    float32
[TL]   layer   5: h2/bn/leaky_relu:0   (16, 48, 48, 256)    float32
[TL]   layer   6: h3/c/Conv2D:0        (16, 24, 24, 512)    float32
[TL]   layer   7: h3/bn/leaky_relu:0   (16, 24, 24, 512)    float32
[TL]   layer   8: h4/c/Conv2D:0        (16, 12, 12, 1024)    float32
[TL]   layer   9: h4/bn/leaky_relu:0   (16, 12, 12, 1024)    float32
[TL]   layer  10: h5/c/Conv2D:0        (16, 6, 6, 2048)    float32
[TL]   layer  11: h5/bn/leaky_relu:0   (16, 6, 6, 2048)    float32
[TL]   layer  12: h6/c/Conv2D:0        (16, 6, 6, 1024)    float32
[TL]   layer  13: h6/bn/leaky_relu:0   (16, 6, 6, 1024)    float32
[TL]   layer  14: h7/c/Conv2D:0        (16, 6, 6, 512)    float32
[TL]   layer  15: h7/bn/batchnorm/Add_1:0 (16, 6, 6, 512)    float32
[TL]   layer  16: res/c/Conv2D:0       (16, 6, 6, 128)    float32
[TL]   layer  17: res/bn/leaky_relu:0  (16, 6, 6, 128)    float32
[TL]   layer  18: res/c2/Conv2D:0      (16, 6, 6, 128)    float32
[TL]   layer  19: res/bn2/leaky_relu:0 (16, 6, 6, 128)    float32
[TL]   layer  20: res/c3/Conv2D:0      (16, 6, 6, 512)    float32
[TL]   layer  21: res/bn3/batchnorm/Add_1:0 (16, 6, 6, 512)    float32
[TL]   layer  22: res/add:0            (16, 6, 6, 512)    float32
[TL]   layer  23: ho/flatten:0         (16, 18432)        float32
[TL]   layer  24: ho/dense/bias_add:0  (16, 1)            float32
'''

def Vgg19_simple_api(rgb, reuse):
    """
    Build the VGG 19 Model

    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope("VGG19", reuse=reuse) as vs:
        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0
        # Convert RGB to BGR
        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else:  # TF 1.0
            # print(rgb_scaled)
            red, green, blue = tf.split(rgb_scaled, 3, 3)
             
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        if tf.__version__ <= '0.11':
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
        else:
            bgr = tf.concat(
                [
                    blue - VGG_MEAN[0],
                    green - VGG_MEAN[1],
                    red - VGG_MEAN[2],
                ], axis=3)
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        """ input layer """
        net_in = InputLayer(bgr, name='input')
        """ conv1 """
        network = Conv2d(net_in, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')
        # 最大池化层，取2*2窗口值中的最大值
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
        """ conv2 """
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')
        """ conv3 """
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')
        """ conv4 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')  # (batch_size, 14, 14, 512)
        conv = network      # (16, 28, 28, 512)
        """ conv5 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool5')  # (batch_size, 7, 7, 512)
        """ fc 6~8 """
        network = FlattenLayer(network, name='flatten')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
        # tf.identity：输出等于输入
        network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
        print("build model finished: %fs" % (time.time() - start_time))
        return network, conv
'''
[TL]   layer   0: resize/ResizeBilinear:0 (16, 224, 224, 3)    float32
[TL]   layer   1: conv1_1/Relu:0       (16, 224, 224, 64)    float32
[TL]   layer   2: conv1_2/Relu:0       (16, 224, 224, 64)    float32
[TL]   layer   3: pool1/MaxPool:0      (16, 112, 112, 64)    float32
[TL]   layer   4: conv2_1/Relu:0       (16, 112, 112, 128)    float32
[TL]   layer   5: conv2_2/Relu:0       (16, 112, 112, 128)    float32
[TL]   layer   6: pool2/MaxPool:0      (16, 56, 56, 128)    float32
[TL]   layer   7: conv3_1/Relu:0       (16, 56, 56, 256)    float32
[TL]   layer   8: conv3_2/Relu:0       (16, 56, 56, 256)    float32
[TL]   layer   9: conv3_3/Relu:0       (16, 56, 56, 256)    float32
[TL]   layer  10: conv3_4/Relu:0       (16, 56, 56, 256)    float32
[TL]   layer  11: pool3/MaxPool:0      (16, 28, 28, 256)    float32
[TL]   layer  12: conv4_1/Relu:0       (16, 28, 28, 512)    float32
[TL]   layer  13: conv4_2/Relu:0       (16, 28, 28, 512)    float32
[TL]   layer  14: conv4_3/Relu:0       (16, 28, 28, 512)    float32
[TL]   layer  15: conv4_4/Relu:0       (16, 28, 28, 512)    float32
[TL]   layer  16: pool4/MaxPool:0      (16, 14, 14, 512)    float32
[TL]   layer  17: conv5_1/Relu:0       (16, 14, 14, 512)    float32
[TL]   layer  18: conv5_2/Relu:0       (16, 14, 14, 512)    float32
[TL]   layer  19: conv5_3/Relu:0       (16, 14, 14, 512)    float32
[TL]   layer  20: conv5_4/Relu:0       (16, 14, 14, 512)    float32
[TL]   layer  21: pool5/MaxPool:0      (16, 7, 7, 512)    float32
[TL]   layer  22: flatten:0            (16, 25088)        float32
[TL]   layer  23: fc6/Relu:0           (16, 4096)         float32
[TL]   layer  24: fc7/Relu:0           (16, 4096)         float32
[TL]   layer  25: fc8/bias_add:0       (16, 1000)         float32
'''




def SRGAN_g2(t_image, is_train=False, reuse=False):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)

    96x96 --> 384x384

    Use Resize Conv
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    size = t_image.get_shape().as_list()
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        temp = n

        # B residual blocks
        for i in range(16):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='b_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, name='add3')
        # B residual blacks end

        # n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
        # n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')
        #
        # n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
        # n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')

        ## 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
        n = UpSampling2dLayer(n, size=[size[1] * 2, size[2] * 2], is_scale=False, method=1, align_corners=False, name='up1/upsample2d')
        n = Conv2d(n, 64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=b_init, name='up1/conv2d')  # <-- may need to increase n_filter
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='up1/batch_norm')

        n = UpSampling2dLayer(n, size=[size[1] * 4, size[2] * 4], is_scale=False, method=1, align_corners=False, name='up2/upsample2d')
        n = Conv2d(n, 32, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=b_init, name='up2/conv2d')  # <-- may need to increase n_filter
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='up2/batch_norm')

        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        return n

def SRGAN_d2(t_image, is_train=False, reuse=False):
    """ Discriminator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope("SRGAN_d", reuse=reuse) as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='n64s1/c')

        n = Conv2d(n, 64, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s2/b')

        n = Conv2d(n, 128, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n128s1/b')

        n = Conv2d(n, 128, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n128s2/b')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n256s1/b')

        n = Conv2d(n, 256, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n256s2/b')

        n = Conv2d(n, 512, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n512s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n512s1/b')

        n = Conv2d(n, 512, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n512s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n512s2/b')

        n = FlattenLayer(n, name='f')
        n = DenseLayer(n, n_units=1024, act=lrelu, name='d1024')
        n = DenseLayer(n, n_units=1, name='out')

        logits = n.outputs
        n.outputs = tf.nn.sigmoid(n.outputs)

        return n, logits

# def vgg16_cnn_emb(t_image, reuse=False):
#     """ t_image = 244x244 [0~255] """
#     with tf.variable_scope("vgg16_cnn", reuse=reuse) as vs:
#         tl.layers.set_name_reuse(reuse)
#
#         mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
#         net_in = InputLayer(t_image - mean, name='vgg_input_im')
#         """ conv1 """
#         network = tl.layers.Conv2dLayer(net_in,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 3, 64],  # 64 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv1_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 64, 64],  # 64 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv1_2')
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool1')
#         """ conv2 """
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 64, 128],  # 128 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv2_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 128, 128],  # 128 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv2_2')
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool2')
#         """ conv3 """
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 128, 256],  # 256 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv3_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv3_2')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv3_3')
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool3')
#         """ conv4 """
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 256, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv4_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv4_2')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv4_3')
#
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool4')
#         conv4 = network
#
#         """ conv5 """
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv5_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv5_2')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv5_3')
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool5')
#
#         network = FlattenLayer(network, name='vgg_flatten')
#
#         # # network = DropoutLayer(network, keep=0.6, is_fix=True, is_train=is_train, name='vgg_out/drop1')
#         # new_network = tl.layers.DenseLayer(network, n_units=4096,
#         #                     act = tf.nn.relu,
#         #                     name = 'vgg_out/dense')
#         #
#         # # new_network = DropoutLayer(new_network, keep=0.8, is_fix=True, is_train=is_train, name='vgg_out/drop2')
#         # new_network = DenseLayer(new_network, z_dim, #num_lstm_units,
#         #             b_init=None, name='vgg_out/out')
#         return conv4, network
