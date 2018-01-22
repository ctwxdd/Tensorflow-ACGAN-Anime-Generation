import tensorflow as tf
import tensorflow.contrib as tc
import math
from libs.ops import *

def residual_block(inputs, output_channel, stride, scope, train = True):

    with tf.variable_scope(scope):
        net = conv2(inputs, 3, output_channel, stride, use_bias=False, scope='conv_1')
        net = tf.layers.batch_normalization(net, training=train)
        net = tf.nn.relu(net)
        net = conv2(net, 3, output_channel, stride, use_bias=False, scope='conv_2')
        net = tf.layers.batch_normalization(net, training=train)
        net = net + inputs

    return net

def discriminator_block(inputs, output_channel, kernel_size, stride, scope, update_collection = None,spectural_norm=False):
        res = inputs

        with tf.variable_scope(scope):

            net = conv2d_sn(   inputs,  output_channel, kernel_size, kernel_size, stride, stride, spectral_normed=spectural_norm, update_collection=update_collection, stddev=0.02, name='conv1')
            #net = conv2(inputs, kernel_size, output_channel, stride, use_bias=False, scope='conv1')
            net = lrelu(net, 0.2)
            net = conv2d_sn(net,  output_channel, kernel_size, kernel_size, stride, stride, spectral_normed=spectural_norm, update_collection=update_collection, stddev=0.02, name='conv1')
            #net = conv2(net, kernel_size, output_channel, stride, use_bias=False, scope='conv2')
            net = net + res
            net = lrelu(net, 0.2)

        return net

class Generator_srresnet(object):

    def __init__(self,  
        hidden_size, 
        img_row, 
        img_col, train = True):
        
        self.hidden_size = hidden_size
        self.img_row = img_row
        self.img_col = img_col
        
        self.batch_size = 64
        self.image_size = img_col

        self.num_resblock = 16
        self.train = train

    def __call__(self, tags_vectors, z, reuse=False, train=True, batch_size = 64):

        self.batch_size = batch_size
        self.train = train
        s = self.image_size # output image size [64]

        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
        gf_dim = 64
        c_dim = 3

        w_init = tf.random_normal_initializer(stddev=0.02)
        gamma_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("g_net") as scope:

            if reuse:
                scope.reuse_variables()
            
            noise_vector = tf.concat([z, tags_vectors], axis=1)

            net_h0 = tc.layers.fully_connected(
                noise_vector, 64*s8*s8,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )

            net_h0 = tf.layers.batch_normalization(net_h0, training=train)
            net_h0 = tf.reshape(net_h0, [-1, s8, s8, gf_dim])
            net = tf.nn.relu(net_h0)

            input_stage = net
  
            for i in range(1, self.num_resblock+1 , 1):
                name_scope = 'resblock_%d'%(i)
                net = residual_block(net, 64, 1, name_scope, train=train)


            net =  tf.layers.batch_normalization(net, training=train)
            net = tf.nn.relu(net)

            net = input_stage + net

            net = conv2(net, 3, 256, 1, use_bias=False, scope='conv1')
            net = pixelShuffler(net, scale=2)
            net =  tf.layers.batch_normalization(net, training=train)
            net = tf.nn.relu(net)

            net = conv2(net, 3, 256, 1, use_bias=False, scope='conv2')
            net = pixelShuffler(net, scale=2)
            net =  tf.layers.batch_normalization(net, training=train)
            net = tf.nn.relu(net)
            
            net = conv2(net, 3, 256, 1, use_bias=False, scope='conv3')
            net = pixelShuffler(net, scale=2)
            net =  tf.layers.batch_normalization(net, training=train)
            net = tf.nn.relu(net)
   

            net = conv2(net, 9, 3, 1, use_bias=False, scope='conv4')

            net = tf.nn.tanh(net)
            
            return net

    @property
    def vars(self):
        return [var for var in tf.global_variables() if "g_net" in var.name]


class Discriminator_srresnet(object):

    def __init__(self,  
        hidden_size,
        img_row,
        img_col):
        self.hidden_size = hidden_size
        self.img_row = img_row
        self.img_col = img_col
        self.image_size = img_col
    
    def __call__(self, tags_vectors, dis_inputs, reuse=True):

        s = self.image_size # output image size [64]
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
        df_dim = 64
        w_init = tf.random_normal_initializer(stddev=0.02)
        gamma_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("d_net") as scope:

            if reuse:
                scope.reuse_variables()
    
            with tf.variable_scope('input_stage'):
                net = conv2(dis_inputs, 4, 32, 2, scope='conv')
                net = lrelu(net, 0.2)


            res = net
            # The discriminator block part
            # block 1
            net = discriminator_block(net, 32, 3, 1, 'disblock_1')

            # block 2
            net = discriminator_block(net, 32, 3, 1, 'disblock_1_1')


            net = conv2(net, 4, 64, 2, use_bias=False, scope='dis_conv_1')

            # block 3
            net = discriminator_block(net, 64, 3, 1, 'disblock_2_1')
            net = discriminator_block(net, 64, 3, 1, 'disblock_2_2')
            net = discriminator_block(net, 64, 3, 1, 'disblock_2_3')
            net = discriminator_block(net, 64, 3, 1, 'disblock_2_4')

            net = conv2(net, 4, 128, 2, use_bias=False, scope='dis_conv_2')
            net = lrelu(net, 0.2)

            # block 4
            net = discriminator_block(net, 128, 3, 1, 'disblock_3_1')
            net = discriminator_block(net, 128, 3, 1, 'disblock_3_2')
            net = discriminator_block(net, 128, 3, 1, 'disblock_3_3')
            net = discriminator_block(net, 128, 3, 1, 'disblock_3_4')

            net = conv2(net, 3, 256, 2, use_bias=False, scope='dis_conv_3')
            net = lrelu(net, 0.2)


            net = discriminator_block(net, 256, 3, 1, 'disblock_4_1')
            net = discriminator_block(net, 256, 3, 1, 'disblock_4_2')
            net = discriminator_block(net, 256, 3, 1, 'disblock_4_3')
            net = discriminator_block(net, 256, 3, 1, 'disblock_4_4')


            net = conv2(net, 3, 512, 2, use_bias=False, scope='dis_conv_4')
            net = lrelu(net, 0.2)

            net = discriminator_block(net, 512, 3, 1, 'disblock_5_1')
            net = discriminator_block(net, 512, 3, 1, 'disblock_5_2')
            net = discriminator_block(net, 512, 3, 1, 'disblock_5_3')
            net = discriminator_block(net, 512, 3, 1, 'disblock_5_4')

            net = conv2(net, 3, 1024, 2, use_bias=False, scope='dis_conv_5')
            net = lrelu(net, 0.2)

            net = tf.reshape(net, [-1, 2 * 2 * 1024])

            with tf.variable_scope('dense_layer_1'):
                net_class = denselayer(net, 23)

                net_class = tf.reshape(net_class, [-1, 23])

            with tf.variable_scope('dense_layer_2'):
                net = denselayer(net, 1)

        return net, net_class

    @property
    def vars(self):
        return [var for var in tf.global_variables() if "d_net" in var.name]

class Discriminator_srresnet_sn(object):

    def __init__(self,  
        hidden_size,
        img_row,
        img_col):
        self.hidden_size = hidden_size
        self.img_row = img_row
        self.img_col = img_col
        self.image_size = img_col
        
    
    def __call__(self, tags_vectors, dis_inputs, reuse=True, update_collection = None):

        s = self.image_size # output image size [64]
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
        df_dim = 64
        w_init = tf.random_normal_initializer(stddev=0.02)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        sn = True

        update_collection = None

        with tf.variable_scope("d_net") as scope:
            
            if reuse:
                scope.reuse_variables()
    
            with tf.variable_scope('input_stage'):
                net = conv2(dis_inputs, 4, 32, 2, scope='conv')
                net = lrelu(net, 0.2)


            res = net
            # The discriminator block part
            # block 1
            net = discriminator_block(net, 32, 3, 1, 'disblock_1', spectural_norm= sn, update_collection= update_collection)

            # block 2
            net = discriminator_block(net, 32, 3, 1, 'disblock_1_1', spectural_norm=sn, update_collection= update_collection)


            #net = conv2(net, 4, 64, 2, use_bias=False, scope='dis_conv_1')
            net = conv2d_sn(   net,  64, 4, 4, 2, 2, spectral_normed=sn, update_collection=update_collection, stddev=0.02, name='dis_conv_1')

            # block 3
            net = discriminator_block(net, 64, 3, 1, 'disblock_2_1', spectural_norm=sn, update_collection= update_collection)
            net = discriminator_block(net, 64, 3, 1, 'disblock_2_2', spectural_norm=sn, update_collection= update_collection)
            net = discriminator_block(net, 64, 3, 1, 'disblock_2_3', spectural_norm=sn, update_collection= update_collection)
            net = discriminator_block(net, 64, 3, 1, 'disblock_2_4', spectural_norm=sn, update_collection= update_collection)

            #net = conv2(net, 4, 128, 2, use_bias=False, scope='dis_conv_2')
            net = conv2d_sn(   net,  128, 4, 4, 2, 2, spectral_normed=sn, update_collection=update_collection, stddev=0.02, name='dis_conv_2')
            net = lrelu(net, 0.2)

            # block 4
            net = discriminator_block(net, 128, 3, 1, 'disblock_3_1', spectural_norm=sn, update_collection= update_collection)
            net = discriminator_block(net, 128, 3, 1, 'disblock_3_2', spectural_norm=sn, update_collection= update_collection)
            net = discriminator_block(net, 128, 3, 1, 'disblock_3_3', spectural_norm=sn, update_collection= update_collection)
            net = discriminator_block(net, 128, 3, 1, 'disblock_3_4', spectural_norm=sn, update_collection= update_collection)

            #net = conv2(net, 3, 256, 2, use_bias=False, scope='dis_conv_3')
            net = conv2d_sn(   net,  256, 3, 3, 2, 2, spectral_normed=sn, update_collection=update_collection, stddev=0.02, name='dis_conv_3')
            net = lrelu(net, 0.2)


            net = discriminator_block(net, 256, 3, 1, 'disblock_4_1', spectural_norm=sn, update_collection= update_collection)
            net = discriminator_block(net, 256, 3, 1, 'disblock_4_2', spectural_norm=sn, update_collection= update_collection)
            net = discriminator_block(net, 256, 3, 1, 'disblock_4_3', spectural_norm=sn, update_collection= update_collection)
            net = discriminator_block(net, 256, 3, 1, 'disblock_4_4', spectural_norm=sn, update_collection= update_collection)


            #net = conv2(net, 3, 512, 2, use_bias=False, scope='dis_conv_4')
            net = conv2d_sn(   net,  512, 3, 3, 2, 2, spectral_normed=sn, update_collection=update_collection, stddev=0.02, name='dis_conv_4')
            net = lrelu(net, 0.2)

            net = discriminator_block(net, 512, 3, 1, 'disblock_5_1', spectural_norm=sn, update_collection= update_collection)
            net = discriminator_block(net, 512, 3, 1, 'disblock_5_2', spectural_norm=sn, update_collection= update_collection)
            net = discriminator_block(net, 512, 3, 1, 'disblock_5_3', spectural_norm=sn, update_collection= update_collection)
            net = discriminator_block(net, 512, 3, 1, 'disblock_5_4', spectural_norm=sn, update_collection= update_collection)

            #net = conv2(net, 3, 1024, 2, use_bias=False, scope='dis_conv_5')
            net = conv2d_sn(net,  1024, 3, 3, 2, 2, spectral_normed=sn, update_collection=update_collection, stddev=0.02, name='dis_conv_5')
            net = lrelu(net, 0.2)

            net = tf.reshape(net, [-1, 2 * 2 * 1024])

            with tf.variable_scope('dense_layer_1'):
                #net_class = denselayer(net, 23)
                net_class = net = linear(net, 23, spectral_normed=sn, update_collection=update_collection, stddev=0.02, name='dense_layer_1')

                net_class = tf.reshape(net_class, [-1, 23])

            with tf.variable_scope('dense_layer_2'):
                #net = denselayer(net, 1)
                net = linear(net, 1, spectral_normed=sn, update_collection=update_collection, stddev=0.02, name='dense_layer_2')


        return net, net_class

    @property
    def vars(self):
        return [var for var in tf.global_variables() if "d_net" in var.name]

    
