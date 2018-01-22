import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
import time
import os
from network.model_srresnet import Generator_srresnet, Discriminator_srresnet_sn
import libs.util as util

class ACGAN(object):
    
    def __init__(self, data, FLAGS):
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config = config)
        self.data = data
        self.tag_len = data.e_len + data.h_len
        self.FLAGS = FLAGS
        self.z_dim = FLAGS.z_dim
        self.img_row = self.data.img_feat.shape[1]
        self.img_col = self.data.img_feat.shape[2]
        self.alpha = 10.
        self.batch_size = self.FLAGS.batch_size
        self.gen_path()
        self.lambd = 0.5
        self.la = 5
        self.logs_path = './logs'
        self.log = True

    def gen_path(self):
        
        # Output directory for models and summaries
        timestamp = str(time.strftime('%b-%d-%Y-%H-%M-%S'))
        self.out_dir = os.path.abspath(os.path.join(os.path.curdir, "models", timestamp))
        print ("Writing to {}\n".format(self.out_dir))

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        self.checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model")
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def get_random_tag(self):

        eye_label = np.random.randint(0, self.data.e_len, self.batch_size)
        hair_label = np.random.randint(0, self.data.h_len, self.batch_size)
        random_tag = np.zeros((self.batch_size, self.tag_len))
        random_tag[np.arange(self.batch_size), eye_label] = 1
        random_tag[np.arange(self.batch_size), hair_label + self.data.e_len] = 1

        return random_tag

    def build_model(self):

        self.g_net = Generator_srresnet(    
                        hidden_size=100,
                        img_row=self.img_row,
                        img_col=self.img_col)

        self.d_net = Discriminator_srresnet_sn(  
                        hidden_size=100,
                        img_row=self.img_row,
                        img_col=self.img_col)

        self.seq = tf.placeholder(tf.float32, [None, self.tag_len], name="seq")
        self.img = tf.placeholder(tf.float32, [self.batch_size, self.img_row, self.img_col, 3], name="img")
        self.rand_seq = tf.placeholder(tf.float32, [None, self.tag_len], name="rand_seq")

        self.z = tf.placeholder(tf.float32, [None, self.z_dim])

        #train with real
        pred_real, pred_real_tag = self.d_net(self.seq, self.img, reuse=False, update_collection = None)
        loss_d_real_label = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_real, labels=tf.ones_like(pred_real)))
        loss_d_real_tag = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_real_tag, labels=self.seq),axis=1))
        loss_d_real = self.la * loss_d_real_label + loss_d_real_tag

        #train with fake
        d_f_tag = self.rand_seq
        fake = self.g_net(d_f_tag, self.z)
        pred_fake, pred_fake_tag = self.d_net(d_f_tag, fake, update_collection = "NO_OPS")

        loss_d_fake_label = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_fake, labels=tf.zeros_like(pred_fake)))
        loss_d_fake_tag = tf.reduce_mean(tf.reduce_sum ( tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_fake_tag, labels=d_f_tag), axis=1))
        loss_d_fake = self.la * loss_d_fake_label #+ loss_d_fake_tag

        self.d_loss = loss_d_real + loss_d_fake 
        #update generator
        tags = self.rand_seq
        self.generated= self.g_net(tags, self.z, reuse=True)

        d_g, d_gc = self.d_net(tags, self.generated, update_collection = "NO_OPS")
        loss_g_label = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_g, labels=tf.ones_like(d_g)))
        loss_g_tag = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_gc, labels=tags), axis=1))
        
        self.g_loss = self.la * loss_g_label + loss_g_tag


        self.loss_cls_d = loss_d_real_tag
        self.loss_cls_g = loss_g_tag

        self.global_step = tf.Variable(0, name='g_global_step', trainable=False)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

            self.d_updates = tf.train.AdamOptimizer(self.FLAGS.lr, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=self.d_net.vars)
            self.g_updates = tf.train.AdamOptimizer(self.FLAGS.lr, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=self.g_net.vars, global_step=self.global_step)

        self.sampler = tf.identity(self.g_net(self.seq, self.z, reuse=True, train=False), name='sampler') 
        self.img_feats = (self.sampler + 1.)/2 * 255
        self.summaries = tf.summary.image('name', self.img_feats,  max_outputs=10)

        if self.log:
            self.summary_writer = tf.summary.FileWriter(self.logs_path,graph=tf.get_default_graph())

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())

        ##load the checkpoint

        #self.saver.restore(self.sess, './models/Jan-02-2018-08-53-53/checkpoints/model-352500')


    def train(self):

        batch_num = self.data.length//self.FLAGS.batch_size if self.data.length%self.FLAGS.batch_size==0 else self.data.length//self.FLAGS.batch_size + 1
        
        print("Start training ACGAN\n")

        for t in range(self.FLAGS.iter):

            img, tags = self.data.next_data_batch(self.FLAGS.batch_size)

            z = self.data.next_noise_batch(len(tags), self.FLAGS.z_dim)

            feed_dict = {
                self.seq:tags,
                self.img:img,
                self.rand_seq: self.get_random_tag(),
                self.z:z,
            }

            _, loss = self.sess.run([self.d_updates, self.d_loss], feed_dict=feed_dict)
            d_cost = loss

            z = self.data.next_noise_batch(len(tags), self.FLAGS.z_dim)
            
            feed_dict = {
                self.img:img,
                self.rand_seq: self.get_random_tag(),
                self.seq:tags,
                self.z:z
            }

            _, loss, c_loss, c_g_loss, step = self.sess.run([self.g_updates, self.g_loss, self.loss_cls_d, self.loss_cls_g, self.global_step], feed_dict=feed_dict)

            current_step = tf.train.global_step(self.sess, self.global_step)

            g_cost = loss

            if current_step % self.FLAGS.display_every == 0:

                print("Epoch {}, Current_step {}".format(self.data.epoch, current_step))
                print("Discriminator loss :{}".format(d_cost))
                print("Generator loss     :{}".format(g_cost))
                print("Cls loss     :{}".format(c_loss))
                print("Cls g loss     :{}".format(c_g_loss))
                print("---------------------------------")

            if current_step % self.FLAGS.checkpoint_every == 0:

                path = self.saver.save(self.sess, self.checkpoint_prefix, global_step=current_step)
                print ("\nSaved model checkpoint to {}\n".format(path))

            if current_step % self.FLAGS.dump_every == 0:
                self.eval(current_step)
                print("Dump test image")

    def eval(self, iters):
        
        z = self.data.fixed_z       
        feed_dict = {
            self.seq:self.data.test_tags_idx,
            self.z:z
        }

        if self.log:
             summ = self.sess.run(self.summaries,feed_dict)
             self.summary_writer.add_summary(summ, iters)
        else:
            f_imgs = self.sess.run(self.sampler, feed_dict=feed_dict)
            util.dump_img(self.FLAGS.img_dir, f_imgs, iters)


