from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Flatten, Dense, Input, Conv2D
from keras.models import Model
import scipy.signal
os.environ["KERAS_BACKEND"] = "tensorflow"


class QlearningAgent:
    def __init__(self, session, action_size, h, w, channels, opt=tf.train.AdamOptimizer(1e-4), gamma=0.99, ):

        """Creates Q-Learning agent
        :param session: tensorflow session
        :param action_size: (int) length of action space
        :param h: (int) input image height
        :param w: (int) input image width
        :param channels: (int) number of image channels
        :param opt: tensorflow optimizer (by default: Adam optimizer)"""
        self.gamma = gamma
        self.action_size = action_size
        self.opt = opt
        self.global_step = tf.Variable(0, name='frame', trainable=False)
        self.frame_inc_op = self.global_step.assign_add(1, use_locking=True)
        K.set_session(session)
        self.sess = session
        with tf.variable_scope('network'):
            self.action = tf.placeholder('int32', [None], name='action')
            self.reward_nstep_q = tf.placeholder('float32', [None], name='reward')
            self.reward_r = tf.placeholder('float32', [None], name='reward_r')
            q_model, r_model, u_model, self.state, self.q_values, self.r_values, self.u_values \
                = self._build_model(h, w, channels)
            self.q_weights = q_model.trainable_weights
            self.r_weights = r_model.trainable_weights
            self.u_weights = u_model.trainable_weights
        with tf.variable_scope('optimizer'):
            # Zero all actions, except one that was performed
            action_onehot = tf.one_hot(self.action, self.action_size, 1.0, 0.0)
            # Predict immediate reward for performed action
            r_value = tf.reduce_sum(tf.multiply(self.r_values, action_onehot), reduction_indices=1)
            # predict expected future uncertainty for performed action
            u_value = tf.reduce_sum(tf.multiply(self.u_values, action_onehot), reduction_indices=1)
            # Predict expected future reward for performed action
            q_value = tf.reduce_sum(tf.multiply(self.q_values, action_onehot), reduction_indices=1)

            # Calculate immediate reward errors
            self.immediate_reward_err = tf.square(self.reward_r - r_value)

            self.u_target = tf.placeholder('float32', self.immediate_reward_err.shape, name='for_u_t')

            self.r_loss = tf.reduce_mean(self.immediate_reward_err)
            self.u_loss = tf.reduce_mean(tf.square(self.u_target - u_value))
            self.q_loss = tf.reduce_mean(tf.square(self.reward_nstep_q - q_value))
            # Define squared mean loss function: (y - y_)^2
            self.loss = 1.0 * self.q_loss + 1.0 * self.r_loss + 1.0 * self.u_loss
            # Compute gradients w.r.t. weights
            grads = tf.gradients(self.loss, tf.trainable_variables())
            # grads = tf.gradients(self.loss, [self.q_weights, self.r_weights, self.u_weights])
            # Apply gradient norm clipping
            grads, _ = tf.clip_by_global_norm(grads, 40.)
            grads_vars = list(zip(grads, tf.trainable_variables()))
            self.train_op = opt.apply_gradients(grads_vars)
        with tf.variable_scope('target_network'):
            target_m, _, _, self.target_state, self.target_q_values, _, _ = self._build_model(h, w, channels)
            target_w = target_m.trainable_weights
        with tf.variable_scope('target_update'):
            self.target_q_update = [target_w[i].assign(self.q_weights[i])
                                  for i in range(len(target_w))]

    def discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
    @property
    def frame(self):
        """:return: global frame"""
        return self.global_step.eval(session=self.sess)

    def update_target(self):
        """Synchronizes shared target with local weights"""
        self.sess.run(self.target_q_update)

    def predict_rewards(self, state):
        """Predicts reward per action for given state.
        :param state: array with shape=[batch_size, num_channels, width, height]
        :type state: nd.array
        :return: rewards for each action (e.g. [1.2, 5.0, 0.4])
        :rtype: list"""
        return self.sess.run(self.q_values, {self.state: state}).flatten()

    def predict_uncertainty(self, state):
        return self.sess.run(self.u_values, {self.state: state}).flatten()

    def predict_rewards_n_uncertainty(self, state):
        q_values, u_values = self.sess.run([self.q_values, self.u_values], {self.state: state})
        return q_values, u_values

    def predict_q_u_r_(self, state):
        q_values, u_values, r_values = self.sess.run([self.q_values, self.u_values, self.r_values], {self.state: state})
        return q_values, u_values, r_values

    def get_loss_values(self,state, action, rewards_4q, rewards_4r):
        for_u_t = self.immediate_reward_err.eval(session=self.sess, feed_dict={
            self.state: state,
            self.action: action,
            self.reward_nstep_q: rewards_4q,
            self.reward_r: rewards_4r,
        })
        for_u_t = self.discount(for_u_t, self.gamma)
        q_loss, u_loss, r_loss = self.sess.run([self.q_loss, self.u_loss, self.r_loss], feed_dict={
            self.state: state,
            self.action: action,
            self.reward_nstep_q: rewards_4q,
            self.reward_r: rewards_4r,
            self.u_target: for_u_t
        })
        return q_loss, u_loss, r_loss

    def predict_target(self, state):
        """Predicts maximum action's reward for given state with target network
        :param state: array with shape=[batch_size, num_channels, width, height]
        :type state: nd.array
        :return: maximum expected reward
        :rtype: float"""
        return np.max(self.sess.run(self.target_q_values, {self.target_state: state}).flatten())

    def train(self, states, actions, rewards_4q, rewards_4r):
        """Trains online network on given states and rewards batch
        :param states: batch with screens with shape=[N, H, W, C]
        :param actions: batch with actions indices, e.g. [1, 4, 0, 2]
        :param rewards: batch with received rewards from given actions (e.g. [0.43, 0.5, -0.1, 1.0])
        :type states: nd.array
        :type actions: list
        :type rewards: list"""
        for_u_t = self.immediate_reward_err.eval(session=self.sess, feed_dict={
            self.state: states,
            self.action: actions,
            self.reward_nstep_q: rewards_4q,
            self.reward_r: rewards_4r,
        })
        for_u_t = self.discount(for_u_t, self.gamma)
        self.sess.run(self.train_op, feed_dict={
            self.state: states,
            self.action: actions,
            self.reward_nstep_q: rewards_4q,
            self.reward_r : rewards_4r,
            self.u_target : for_u_t
        })

    def frame_increment(self):
        """Increments global frame counter"""
        self.frame_inc_op.eval(session=self.sess)

    def _build_model(self, h, w, channels, fc3_size=256, ):
        """Builds DQN model (Mnih et al., 2015)
        :param h: input layer height
        :param w: input layer width
        :param channels: input layer number of channels
        :param fc3_size: 3rd fully connected layer size (common: 256, 512)"""

        state = tf.placeholder('float32', shape=(None, h, w, channels), name='state')
        inputs = Input(shape=(h, w, channels,))
        shared_model = Conv2D(activation="relu", filters=16, kernel_size=(8, 8), padding="same",
                       strides=(4, 4), data_format="channels_last")(inputs)
        shared_model = Conv2D(kernel_size=(4, 4), data_format="channels_last", strides=(2, 2),
                       filters=32, activation="relu", padding="same")(shared_model)
        layer_3 = Flatten()(shared_model)

        r_layer1 = Dense(units=512, activation='relu')(layer_3)
        r_layer2 = Dense(units=256, activation='relu')(r_layer1)
        r_out = Dense(units=self.action_size, activation='linear')(r_layer2)
        r_model = Model(inputs=inputs, outputs=r_out)
        rvalues = r_model(state)

        u_layer1 = Dense(units=512, activation='relu')(layer_3)
        u_layer2 = Dense(units=256, activation='relu')(u_layer1)
        u_out = Dense(units=self.action_size, activation='linear')(u_layer2)
        u_model = Model(inputs=inputs, outputs=u_out)
        uvalues = u_model(state)

        q_layer1 = Dense(units=512, activation='relu')(layer_3)
        q_layer2 = Dense(units=256, activation='relu')(q_layer1)
        q_out = Dense(units=self.action_size, activation='linear')(q_layer2)
        q_model = Model(inputs=inputs, outputs=q_out)
        qvalues = q_model(state)

        return q_model, r_model, u_model, state, qvalues, rvalues, uvalues


class AgentSummary:
    """Helper wrapper for summary tensorboard logging"""

    def __init__(self, logdir, agent, env_name):
        """ :param logdir: path to the log directory
            :param agent: agent class-wrapper
            :param env_name: environment name"""
        with tf.variable_scope('summary'):
            self.agent = agent
            self.last_time = time.time()
            self.last_frames = self.agent.frame
            scalar_tags = ['fps', 'episode_avg_reward', 'avg_q_value', 'avg_u_value', 'avg_q_loss',
                    'avg_u_loss', 'avg_r_loss', 'epsilon', 'total_frame_step']
            self.writer = tf.summary.FileWriter(logdir, self.agent.sess.graph)
            self.summary_vars = {}
            self.summary_ph = {}
            self.summary_ops = {}
            for k in scalar_tags:
                self.summary_vars[k] = tf.Variable(0.)
                self.summary_ph[k] = tf.placeholder('float32', name=k)
                self.summary_ops[k] = tf.summary.scalar("%s/%s" % (env_name, k), self.summary_vars[k])
            self.update_ops = []
            for k in self.summary_vars:
                self.update_ops.append(self.summary_vars[k].assign(self.summary_ph[k]))
            self.summary_op = tf.summary.merge(list(self.summary_ops.values()))

    def write_summary(self, tags):
        """Writes summary to TensorBoard.
        :param tags: summary dictionary with with keys:
                     'episode_avg_reward': average episode reward;
                     'avg_q_value'       : average episode Q-value;
                     'avg_u_value'       : average episode U-value;
                     'epsilon'           : current epsilon values;
                     'total_frame_step'  : current frame step.
        :type tags: dict"""
        tags['fps'] = (self.agent.frame - self.last_frames) / (time.time() - self.last_time)
        self.last_time = time.time()
        self.last_frames = self.agent.frame
        self.agent.sess.run(self.update_ops, {self.summary_ph[k]: v for k, v in tags.items()})
        summary = self.agent.sess.run(self.summary_op,
                                      {self.summary_vars[k]: v for k, v in tags.items()})
        self.writer.add_summary(summary, global_step=self.agent.frame)
