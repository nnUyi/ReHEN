# -*- coding=utf-8 -*-

'''
	author: Youzhao Yang
	date: 12/25/2019
	github: https://github.com/nnuyi
'''

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import time
import cv2

# customer libraries
from utils import save_images, read_data
from metrics import SSIM, PSNR
from settings import *

from ops import bn, UPDATE_G_OPS_COLLECTION

class DerainNet:
    model_name = 'ReHEN'
    
    '''Derain Net: all the implemented layer are included (e.g. SEBlock,
                                                                HEU,
                                                                REU,
                                                                ReHEB).
        Params:
            config: the training configuration
            sess: runing session
    '''
    
    def __init__(self, config, sess=None):
        # config proto
        self.config = config
        self.channel_dim = self.config.channel_dim
        self.batch_size = self.config.batch_size
        self.patch_size = self.config.patch_size
        self.input_channels = self.config.input_channels
        
        # metrics
        self.ssim = SSIM(max_val=1.0)
        self.psnr = PSNR(max_val=1.0)

        # create session
        self.sess = sess
    
    # global average pooling
    def globalAvgPool2D(self, input_x):
        global_avgpool2d = tf.contrib.keras.layers.GlobalAvgPool2D()
        return global_avgpool2d(input_x)
    
    # leaky relu
    def leakyRelu(self, input_x):
        leaky_relu = tf.contrib.keras.layers.LeakyReLU(alpha=0.2)
        return leaky_relu(input_x)

    # squeeze-and-excitation block
    def SEBlock(self, input_x, input_dim=32, reduce_dim=8, scope='SEBlock'):
        with tf.variable_scope(scope) as scope:
            # global scale
            global_pl = self.globalAvgPool2D(input_x)
            reduce_fc1 = slim.fully_connected(global_pl, reduce_dim, activation_fn=tf.nn.relu)
            reduce_fc2 = slim.fully_connected(reduce_fc1, input_dim, activation_fn=None)
            g_scale = tf.nn.sigmoid(reduce_fc2)
            g_scale = tf.expand_dims(g_scale, axis=1)
            g_scale = tf.expand_dims(g_scale, axis=1)
            gs_input = input_x*g_scale
            return gs_input

    # recurrent enhancement unit
    def REU(self, input_x, h, out_dim, scope='REU'):
        with tf.variable_scope(scope):
            if h is None:
                self.conv_xz = slim.conv2d(input_x, out_dim, 3, 1, scope='conv_xz')
                self.conv_xn = slim.conv2d(input_x, out_dim, 3, 1, scope='conv_xn')
                z = tf.nn.sigmoid(self.conv_xz)
                f = tf.nn.tanh(self.conv_xn)
                h = z*f
            else:
                self.conv_hz = slim.conv2d(h, out_dim, 3, 1, scope='conv_hz')
                self.conv_hr = slim.conv2d(h, out_dim, 3, 1, scope='conv_hr')

                self.conv_xz = slim.conv2d(input_x, out_dim, 3, 1, scope='conv_xz')
                self.conv_xr = slim.conv2d(input_x, out_dim, 3, 1, scope='conv_xr')
                self.conv_xn = slim.conv2d(input_x, out_dim, 3, 1, scope='conv_xn')
                r = tf.nn.sigmoid(self.conv_hr+self.conv_xr)
                z = tf.nn.sigmoid(self.conv_hz+self.conv_xz)
                
                self.conv_hn = slim.conv2d(r*h, out_dim, 3, 1, scope='conv_hn')
                n = tf.nn.tanh(self.conv_xn + self.conv_hn)
                h = (1-z)*h + z*n

        # channel attention block
        se = self.SEBlock(h, out_dim, reduce_dim=int(out_dim/4))
        h = self.leakyRelu(se)
        return h, h

    # hierarchy enhancement unit
    def HEU(self, input_x, is_training=False, scope='HEU'):
        with tf.variable_scope(scope) as scope:
            local_shortcut = input_x
            dense_shortcut = input_x
            
            for i in range(1, 3):
                with tf.variable_scope('ResBlock_{}'.format(i)):
                    with tf.variable_scope('Conv1'):
                        conv_tmp1 = slim.conv2d(local_shortcut, self.channel_dim,3,1)
                        conv_tmp1_bn = bn(conv_tmp1, is_training, UPDATE_G_OPS_COLLECTION)
                        out_tmp1 = tf.nn.relu(conv_tmp1_bn)

                    with tf.variable_scope('Conv2'):
                        conv_tmp2 = slim.conv2d(out_tmp1, self.channel_dim,3,1)
                        conv_tmp2_bn = bn(conv_tmp2, is_training, UPDATE_G_OPS_COLLECTION)
                        out_tmp2 = tf.nn.relu(conv_tmp2_bn)
                        conv_shortcut = tf.add(local_shortcut, out_tmp2)

                dense_shortcut = tf.concat([dense_shortcut, conv_shortcut], -1)
                local_shortcut = conv_shortcut

            with tf.variable_scope('Trans'):
                conv_tmp3 = slim.conv2d(dense_shortcut, self.channel_dim, 3,1)
                conv_tmp3_bn = bn(conv_tmp3, is_training, UPDATE_G_OPS_COLLECTION)
                conv_tmp3_se = self.SEBlock(conv_tmp3_bn, self.channel_dim, reduce_dim=int(self.channel_dim/4))
                out_tmp3 = tf.nn.relu(conv_tmp3_se)
                heu_f = tf.add(input_x, out_tmp3)

            return heu_f

    # recurrent hierarchy enhancement block
    def ReHEB(self, input_x, h, is_training=False, scope='ReHEB'):
        with tf.variable_scope(scope):
            if input_x.get_shape().as_list()[-1] == 3:
                heu = input_x
            else:
                heu = self.HEU(input_x, is_training=is_training)
            reheb, h = self.REU(heu, h, out_dim=self.channel_dim)
        return reheb, h

    # recurrent hierarchy and enhancement network
    def derainNet(self, input_x, is_training=False, scope_name='derainNet'):
        '''ReHEN: recurrent hierarchy and enhancement network
            Params:
                input_x: input data
                is_training: training phase or testing phase
                scope_name: the scope name of the ReHEN (customer definition, default='derainNet')
            Return:
                return the derained results

            Input shape:
                4D tensor with shape '(batch_size, height, width, channels)'
                
            Output shape:
                4D tensor with shape '(batch_size, height, width, channels)'
        '''
        # reuse: tf.AUTO_REUSE(such setting will enable the network to reuse parameters automatically)
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            # convert is_training variable to tensor type
            is_training = tf.convert_to_tensor(is_training, dtype='bool', name='is_training')            
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], 
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              normalizer_fn=None,
                                              activation_fn=None,
                                              padding='SAME'):
                
                stages = 4
                block_num = 5
                old_states = [None for _ in range(block_num)]
                oups = []
                ori = input_x
                shallow_f = input_x

                for stg in range(stages):
                    # recurrent hierarchy enhancement block (ReHEB)
                    with tf.variable_scope('ReHEB'):
                        states = []
                        for i in range(block_num):
                            sp = 'ReHEB_{}'.format(i)
                            shallow_f, st = self.ReHEB(shallow_f, old_states[i], is_training=is_training, scope=sp)
                            states.append(st)

                    further_f = shallow_f

                    # residual map generator (RMG)
                    with tf.variable_scope('RMG'):
                        rm_conv = slim.conv2d(further_f, self.channel_dim, 3, 1)
                        rm_conv_se = self.SEBlock(rm_conv, self.channel_dim, reduce_dim=int(self.channel_dim/4))
                        rm_conv_a = self.leakyRelu(rm_conv_se)
                        neg_residual_conv = slim.conv2d(rm_conv_a, self.input_channels, 3, 1)
                        neg_residual = neg_residual_conv
                    shallow_f = ori - neg_residual
                    oups.append(shallow_f)
                    old_states = [tf.identity(s) for s in states]
                    
        return oups, shallow_f, neg_residual
    
    def build(self):
        # placeholder
        self.rain = tf.placeholder(tf.float32, [None, None, None, self.input_channels], name='rain')
        self.norain = tf.placeholder(tf.float32, [None, None, None, self.input_channels], name='norain')
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        
        # derainnet
        self.oups, self.out, self.residual = self.derainNet(self.rain, is_training=self.config.is_training)
        self.finer_out = tf.clip_by_value(self.out, 0, 1.0)
        self.finer_residual = tf.clip_by_value(tf.abs(self.residual), 0, 1)
        
        # metrics
        self.ssim_finer_tensor = tf.reduce_mean(self.ssim._ssim(self.norain, self.out, 0, 0))
        self.psnr_finer_tensor = tf.reduce_mean(self.psnr.compute_psnr(self.norain, self.out))
        self.ssim_val = tf.reduce_mean(self.ssim._ssim(self.norain, self.finer_out, 0, 0))
        self.psnr_val = tf.reduce_mean(self.psnr.compute_psnr(self.norain, self.finer_out))
        
        # loss function
        # MSE loss
        self.l2_loss = tf.reduce_sum([tf.reduce_mean(tf.square(out - self.norain)) for out in self.oups])
        # SSIM loss
        self.ssim_loss = tf.log(1.0/(self.ssim_finer_tensor+1e-5))
        # PSNR loss
        self.psnr_loss = 1.0/(self.psnr_finer_tensor + 1e-3)
        # total loss
        self.total_loss = self.l2_loss + 0.001*self.ssim_loss + 0.1*self.psnr_loss
        
        # optimization
        t_vars = tf.trainable_variables()
        g_vars = [var for var in t_vars if 'derainNet' in var.name]
        loss_train_ops = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.config.beta1, beta2=self.config.beta2).minimize(self.total_loss, var_list=g_vars)

        # batchnorm training ops
        batchnorm_ops = tf.get_collection(UPDATE_G_OPS_COLLECTION)
        bn_update_ops = tf.group(*batchnorm_ops)
        self.train_ops = tf.group(loss_train_ops, bn_update_ops)

        # summary
        self.l2_loss_summary = tf.summary.scalar('l2_loss', self.l2_loss)
        self.total_loss_summary = tf.summary.scalar('total_loss', self.total_loss)
        self.edge_loss_summary = tf.summary.scalar('ssim_loss', self.ssim_loss)
        self.edge_loss_summary = tf.summary.scalar('psnr_loss', self.psnr_loss)
        self.ssim_summary = tf.summary.scalar('ssim', self.ssim_val)
        self.psnr_summary = tf.summary.scalar('psnr', self.psnr_val)
        self.summaries = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.config.logs_dir, self.sess.graph)
        
        # saver
        global_variables = tf.global_variables()
        var_to_store = [var for var in global_variables if 'derainNet' in var.name]
        self.saver = tf.train.Saver(var_list=var_to_store)

        # trainable variables
        num_params = 0
        for var in g_vars:
            tmp_num = 1
            for i in var.get_shape().as_list():
                tmp_num = tmp_num*i
            num_params = num_params + tmp_num
        print('numbers of trainable parameters:{}'.format(num_params))

    # training phase
    def train(self):
        # initialize variables
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # load training model
        check_bool = self.load_model()
        if check_bool:
            print('[!!!] load model successfully')
        else:
            print('[***] fail to load model')
        
        lr_ = self.config.lr
        start_time = time.time()
        for counter in range(self.config.iterations):
            if counter == 50000:
                lr_ = 0.1*lr_

            # obtain training image pairs
            img, label = read_data(self.config.train_dataset, self.config.data_path, self.batch_size, self.patch_size, self.config.trainset_size)
            _, total_loss, summaries, ssim, psnr = self.sess.run([self.train_ops,
                                                               self.total_loss,
                                                               self.summaries,
                                                               self.ssim_val,
                                                               self.psnr_val], feed_dict={self.rain:img,
                                                                                           self.norain:label,
                                                                                           self.lr:lr_})

            print('Iteration:{}, phase:{}, loss:{:.4f}, ssim:{:.4f}, psnr:{:.4f}, lr:{}, iterations:{}'.format(counter,
                                                                                                                 self.config.phase,
                                                                                                                 total_loss,
                                                                                                                 ssim,
                                                                                                                 psnr,
                                                                                                                 lr_,
                                                                                                                 self.config.iterations))
                                
            self.summary_writer.add_summary(summaries, global_step=counter)
            if np.mod(counter, 100)==0:
                self.sample(self.config.sample_dir, counter)

            if np.mod(counter, 500)==0:
                self.save_model()
        
        # save final model
        if counter == self.config.iterations-1:
            self.save_model()

        # training time
        end_time = time.time()
        print('training time:{} hours'.format((end_time-start_time)/3600.0))

    # sampling phase
    def sample(self, sample_dir, iterations):
        # obtaining sampling image pairs
        test_img, test_label = read_data(self.config.test_dataset, self.config.data_path, self.batch_size, self.patch_size, self.config.testset_size)
        finer_out, finer_residual = self.sess.run([self.finer_out, self.finer_residual], feed_dict={self.rain:test_img})
        
        # save sampling images
        test_img_uint8 = np.uint8(test_img*255.0)
        test_label_uint8 = np.uint8(test_label*255.0)
        finer_out_uint8 = np.uint8(finer_out*255.0)
        finer_residual = np.uint8(finer_residual*255.0)
        sample = np.concatenate([test_img_uint8, test_label_uint8, finer_out_uint8, finer_residual], 2)
        save_images(sample, [int(np.sqrt(self.batch_size))+1, int(np.sqrt(self.batch_size))+1], '{}/{}_{}_{:04d}.jpg'.format(self.config.sample_dir,
                                                                                                                             self.config.test_dataset,
                                                                                                                             self.config.phase,
                                                                                                                             iterations))
    
    # testing phase
    def test(self):
        rain = tf.placeholder(tf.float32, [None, None, None, self.input_channels], name='test_rain')
        norain = tf.placeholder(tf.float32, [None, None, None, self.input_channels], name='test_norain')
        
        oups, out, residual = self.derainNet(rain, is_training=self.config.is_training)
        finer_out = tf.clip_by_value(out, 0, 1.0)
        finer_residual = tf.clip_by_value(tf.abs(residual), 0, 1.0)

        ssim_val = tf.reduce_mean(self.ssim._ssim(norain, finer_out, 0, 0))
        psnr_val = tf.reduce_mean(self.psnr.compute_psnr(norain, finer_out))

        # load model
        self.saver = tf.train.Saver()
        check_bool = self.load_model()
        if check_bool:
            print('[!!!] load model successfully')
        else:
            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()
            print('[***] fail to load model')

        try:
            test_num, test_data_format, test_label_format = test_dic[self.config.test_dataset]
        except:
            print('no testing dataset named {}'.format(self.config.test_dataset))
            return

        ssim = []
        psnr = []
        for index in range(1, test_num+1):
            test_data_fn = test_data_format.format(index)
            test_label_fn = test_label_format.format(index)
            
            test_data_path = os.path.join(self.config.test_path.format(self.config.test_dataset), test_data_fn)
            test_label_path = os.path.join(self.config.test_path.format(self.config.test_dataset), test_label_fn)

            test_data_uint8 = cv2.imread(test_data_path)
            test_label_uint8 = cv2.imread(test_label_path)

            test_data_float = test_data_uint8/255.0
            test_label_float = test_label_uint8/255.0
            
            test_data = np.expand_dims(test_data_float, 0)
            test_label = np.expand_dims(test_label_float, 0)
            
            t = 0
            s_t = time.time()
            finer_out_val, finer_residual_val, tmp_ssim, tmp_psnr = self.sess.run([finer_out,
                                                                                   finer_residual,
                                                                                   ssim_val,
                                                                                   psnr_val] , feed_dict={rain:test_data,
                                                                                                          norain:test_label})

            e_t = time.time()            
            total_t = e_t - s_t
            t = t + total_t

            # save psnr and ssim metrics
            ssim.append(tmp_ssim)
            psnr.append(tmp_psnr)
            # save testing image
            test_label = np.uint8(test_label*255)
            finer_out_val = np.uint8(finer_out_val*255)
            finer_residual_val = np.uint8(finer_residual_val*255)
            save_images(finer_out_val, [1,1], '{}/{}_{}'.format(self.config.test_dir, self.config.test_dataset, test_data_fn))
            save_images(test_label, [1,1], '{}/{}'.format(self.config.test_dir, test_data_fn))
            save_images(finer_residual_val, [1,1], '{}/residual_{}'.format(self.config.test_dir, test_data_fn))
            print('test image {}: ssim:{}, psnr:{} time:{:.4f}'.format(test_data_fn, tmp_ssim, tmp_psnr, total_t))
        
        mean_ssim = np.mean(ssim)
        mean_psnr = np.mean(psnr)
        print('Test phase: ssim:{}, psnr:{}'.format(mean_ssim, mean_psnr))
        print('Average time:{}'.format(t/(test_num-1)))

    # save model            
    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.model_name, self.config.train_dataset,
            self.batch_size)
    @property
    def model_pos(self):
        return '{}/{}/{}'.format(self.config.checkpoint_dir, self.model_dir, self.model_name)

    def save_model(self):
        if not os.path.exists(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)
        self.saver.save(self.sess, self.model_pos)
        
    def load_model(self):
        if not os.path.isfile(os.path.join(self.config.checkpoint_dir, self.model_dir,'checkpoint')):
            return False
        else:
            self.saver.restore(self.sess, self.model_pos)
            return True
