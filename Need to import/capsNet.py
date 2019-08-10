import tensorflow as tf

from config import cfg
from utils import get_batch_data
from capsLayer import CapsConv


class CapsNet(object):
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.X, self.Y = get_batch_data()

                self.build_arch()
                self.loss()

                # t_vars = tf.trainable_variables()
                self.optimizer = tf.train.AdamOptimizer()
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)  # var_list=t_vars)
            else:
                self.X = tf.placeholder(tf.float32,
                                        shape=(cfg.batch_size, 28, 28, 1))
                self.build_arch()

        tf.logging.info('Seting up the main structure')

    def build_arch(self):
        with tf.variable_scope('Conv1_layer'):
            # Conv1, [batch_size, 20, 20, 256]
            conv1 = tf.contrib.layers.conv2d(self.X, num_outputs=256,
                                             kernel_size=9, stride=1,
                                             padding='VALID')
            assert conv1.get_shape() == [cfg.batch_size, 20, 20, 256]

        # TODO: Rewrite the 'CapsConv' class as a function, the capsLay
        # function should be encapsulated into tow function, one like conv2d
        # and another is fully_connected in Tensorflow.
        # Primary Capsules, [batch_size, 1152, 8, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsConv(num_units=8, with_routing=False)
            caps1 = primaryCaps(conv1, num_outputs=32, kernel_size=9, stride=2)
            assert caps1.get_shape() == [cfg.batch_size, 1152, 8, 1]

        # DigitCaps layer, [batch_size, 10, 16, 1]
        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsConv(num_units=16, with_routing=True)
            self.caps2 = digitCaps(caps1, num_outputs=10)

        # Decoder structure in Fig. 2
        # 1. Do masking, how:
        with tf.variable_scope('Masking'):
            # a). calc ||v_c||, then do softmax(||v_c||)
            # [batch_size, 10, 16, 1] => [batch_size, 10, 1, 1]
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2),
                                                  axis=2, keep_dims=True))
            self.softmax_v = tf.nn.softmax(self.v_length, dim=1)
            assert self.softmax_v.get_shape() == [cfg.batch_size, 10, 1, 1]

            # b). pick out the index of max softmax val of the 10 caps
            # [batch_size, 10, 1, 1] => [batch_size] (index)
            argmax_idx = tf.argmax(self.softmax_v, axis=1, output_type=tf.int32)
            assert argmax_idx.get_shape() == [cfg.batch_size, 1, 1]

            # c). indexing
            # It's not easy to understand the indexing process with argmax_idx
            # as we are 3-dim animal
            masked_v = []
            argmax_idx = tf.reshape(argmax_idx, shape=(cfg.batch_size, ))
            for batch_size in range(cfg.batch_size):
                v = self.caps2[batch_size][argmax_idx[batch_size], :]
                masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))

            self.masked_v = tf.concat(masked_v, axis=0)
            assert self.masked_v.get_shape() == [cfg.batch_size, 1, 16, 1]

        # 2. Reconstructe the MNIST images with 3 FC layers
        # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
        with tf.variable_scope('Decoder'):
            vector_j = tf.reshape(self.masked_v, shape=(cfg.batch_size, -1))
            fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
            assert fc1.get_shape() == [cfg.batch_size, 512]
            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
            assert fc2.get_shape() == [cfg.batch_size, 1024]
            self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)

    def loss(self):
        # 1. The margin loss

        # [batch_size, 10, 1, 1]
        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., cfg.m_plus - self.v_length))
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., self.v_length - cfg.m_minus))
        assert max_l.get_shape() == [cfg.batch_size, 10, 1, 1]

        # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))

        # calc T_c: [batch_size, 10]
        # T_c = Y, is my understanding correct? Try it.
        T_c = self.Y
        # [batch_size, 10], element-wise multiply
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        # 2. The reconstruction loss
        orgin = tf.reshape(self.X, shape=(cfg.batch_size, -1))
        squared = tf.square(self.decoded - orgin)
        self.reconstruction_err = tf.reduce_mean(squared)

        # 3. Total loss
        self.total_loss = self.margin_loss + 0.0005 * self.reconstruction_err

        # Summary
        tf.summary.scalar('margin_loss', self.margin_loss)
        tf.summary.scalar('reconstruction_loss', self.reconstruction_err)
        tf.summary.scalar('total_loss', self.total_loss)
        recon_img = tf.reshape(self.decoded, shape=(cfg.batch_size, 28, 28, 1))
        tf.summary.image('reconstruction_img', recon_img)
        self.merged_sum = tf.summary.merge_all()
