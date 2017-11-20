# based on https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow

import tensorflow as tf
import numpy as np
import ops
import math

init_he = tf.contrib.layers.variance_scaling_initializer(factor=2,    mode='FAN_IN',    uniform=False,    seed=None,  dtype=tf.float32)
init_fn = None
activation_fn = ops.leaky_relu

###############################  DDPG  ####################################
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_high, a_low, lr_a=0.0001, lr_c=0.0003, gamma=0.99, tau=0.001, rpm_size=1000000, batch_size=128):
        self.memory = np.zeros((rpm_size, s_dim * 2 + a_dim + 1 + 1), dtype=np.float32)
        self.rpm_size = rpm_size
        self.batch_size = batch_size
        self.pointer = 0
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.a_high, self.a_low = a_dim, s_dim, a_high,a_low
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.is_done = tf.placeholder(tf.float32, [None, 1], 'is_done')
            
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - tau) * ta + tau * ea), tf.assign(tc, (1 - tau) * tc + tau * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + (1-self.is_done) * gamma * q_

        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(lr_c).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(lr_a).minimize(a_loss, var_list=self.ae_params)

        self.saver = tf.train.Saver(max_to_keep=100)
        
        self.sess.run(tf.global_variables_initializer())
        

    def save(self, step):
        save_path = self.saver.save(self.sess, "./save/model.ckpt", global_step=step)
        print("Model saved in file: {}".format(save_path))

    def load(self):
        save_path = tf.train.latest_checkpoint("./save/");
        if save_path:
            print("Model will loaded from file: {}".format(save_path))
            self.saver.restore(self.sess, save_path )
        else:
            print("no check point file")
    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)
        
        
        if self.pointer < self.rpm_size:
            indices = np.random.choice(self.pointer, size=self.batch_size)
        else:
            indices = np.random.choice(self.rpm_size, size=self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        #br = bt[:, -self.s_dim - 1: -self.s_dim]
        #bs_ = bt[:, -self.s_dim:]
        br = bt[:, self.s_dim + self.a_dim: self.s_dim + self.a_dim + 1]
        bs_ = bt[:, self.s_dim + self.a_dim + 1:self.s_dim + self.a_dim + 1+ self.s_dim]
        bd = bt[:,-1:]
        #print(bd)
        self.sess.run(self.atrain, {self.S: bs, self.is_done:bd})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.is_done:bd})


    def store_transition(self, s, a, r, s_, is_done):
        transition = np.hstack((s, a, [r], s_, [is_done]))
        index = self.pointer % self.rpm_size  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        
    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            
           
                     
            net = tf.layers.dense(s, 128, activation=ops.leaky_relu, name='l1', trainable=trainable, kernel_initializer=init_fn)
            net = tf.layers.dense(net, 128, activation=ops.leaky_relu, name='l2', trainable=trainable,kernel_initializer=init_fn)
            net = tf.layers.dense(net, 256, activation=ops.leaky_relu, name='l3', trainable=trainable,kernel_initializer=init_fn)
            net = tf.layers.dense(net, 128, activation=ops.leaky_relu, name='l4', trainable=trainable,kernel_initializer=init_fn)
            net = tf.layers.dense(net, 128, activation=ops.leaky_relu, name='l5', trainable=trainable,kernel_initializer=init_fn)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable,kernel_initializer=init_fn)
            
            a_multi = (self.a_high-self.a_low)/2.0
            a_shift = (self.a_high+self.a_low)/2.0
            return tf.add(tf.multiply(a, a_multi, name='multi_a'), a_shift, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
             
            h1_size = 128
            h2_size = 128
            h3_size = 128           
            #h1_size = self.s_dim * 10
            #h3_size = 5
            #h2_size = int(math.sqrt(h1_size * h3_size))
            #sa = tf.concat([s,a],1)
            #w1_s = tf.get_variable('w1_s', [self.s_dim, h1_size], trainable=trainable)
            #w1_a = tf.get_variable('w1_a', [self.a_dim, h1_size], trainable=trainable)
            #b1 = tf.get_variable('b1', [1, h1_size], trainable=trainable)
            #net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(s, 128, activation=ops.leaky_relu, name='l1', trainable=trainable,kernel_initializer=init_fn)
            net = tf.layers.dense(net, 128, activation=ops.leaky_relu, name='l2', trainable=trainable,kernel_initializer=init_fn)
            concated = tf.concat([net,a],1)
            net = tf.layers.dense(concated, 256, activation=ops.leaky_relu, name='l3', trainable=trainable,kernel_initializer=init_fn) 
            net = tf.layers.dense(net, 128, activation=ops.leaky_relu, name='l4', trainable=trainable,kernel_initializer=init_fn)
            net = tf.layers.dense(net, 64, activation=ops.leaky_relu, name='l5', trainable=trainable,kernel_initializer=init_fn)
            return tf.layers.dense(net, 1, trainable=trainable, kernel_initializer=init_fn)  # Q(s,a)


