from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import threading as th
import time
from datetime import datetime

from six.moves import range  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf
import scipy.signal
from asyncrl.agent import QlearningAgent, AgentSummary
from asyncrl.environment import GymWrapperFactory

# To reproduce minimum epsilon sampling
random.seed(201)
# Distribution of epsilon exploration chances (0.1 = 0.4; 0.01 = 0.3; 05 = 0.3)
EPS_MIN_SAMPLES = 1 * [0.1] + 1 * [0.01] + 2 * [0.05]

# Configurations
tf.app.flags.DEFINE_integer("threads", 8, "Number of threads to use")
tf.app.flags.DEFINE_boolean("use_cpu", False, "Use CPU or GPU for training (by default is GPU)")
# Training settings
tf.app.flags.DEFINE_boolean("batch_avg_reward", True, "generate reward batch as episodic avg reward for r_fn and u_fn")
tf.app.flags.DEFINE_boolean("ae", False, "autoencoder-not using")
tf.app.flags.DEFINE_integer("total_frames", 50000000, "Total steps (across all threads)")
tf.app.flags.DEFINE_integer("update_interval", 40000, "Update target network after X frames")
tf.app.flags.DEFINE_boolean("eps_use", False, "epsilon-greedy use")
tf.app.flags.DEFINE_float("eps_steps", 4000000.0, "Decrease epsilon over X frames")
tf.app.flags.DEFINE_float("eps_start", 1.0, "Starting epsilon (initial exploration chance)")
tf.app.flags.DEFINE_float("gamma", 0.99, "Gamma discount factor")
tf.app.flags.DEFINE_integer("tmax", 300, "Maximum batch size")
tf.app.flags.DEFINE_integer("action_repeat", 4, "Applies last action to X next frames")
tf.app.flags.DEFINE_integer("memory_len", 4, "Memory length - number of stacked input images")
# Environment settings
tf.app.flags.DEFINE_string("env", 'Freeway-v0', "Environment name (available all OpenAI Gym environments)")
tf.app.flags.DEFINE_boolean("render", False, "Render frames? Significantly slows down training process")
tf.app.flags.DEFINE_integer("width", 84, "Screen image width")
tf.app.flags.DEFINE_integer("height", 84, "Screen image height")
# Logging
tf.app.flags.DEFINE_integer("test_iter", 15, "Number of test iterations. Used for logging.")
tf.app.flags.DEFINE_string("logdir", 'logs_uncertainty/Freeway/111_no_eps_300step_3layers_batch_episodic_avg_reward/', "Path to the directory used for checkpoints and loggings")
# tf.app.flags.DEFINE_string("logdir", 'logs/Frostbite/logs_uncertainty_1.0r_1.0u_1.0q_no_eps_300step_2layers_lin_activ_/', "Path to the directory used for checkpoints and loggings")
tf.app.flags.DEFINE_integer("log_interval", 120000, "Log and checkpoint every X frame")
# Evaluation
tf.app.flags.DEFINE_boolean("eval", False, "Disables training, evaluates agent's performance")
tf.app.flags.DEFINE_string("evaldir", 'eval/', "Path to the evaluation logging")
tf.app.flags.DEFINE_integer("eval_iter", 5, "Number of evaluation episodes")
# Optimizer
tf.app.flags.DEFINE_float("lr", 1e-4, "Starting learning rate")
FLAGS = tf.app.flags.FLAGS
# Hide all GPUs for current process if CPU was chosen
if FLAGS.use_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
global_epsilons = [0.] * FLAGS.threads
training_finished = False


def update_epsilon(frames, eps_steps, eps_min):
    """Anneals epsilon based on current frame
    :param frames: (float) current frame number
    :param eps_steps: (int) total number of epsilon annealing steps
    :param eps_min: (float) minimum allowed epsilon
    :type frames: float
    :type eps_steps: int
    :type eps_min: float"""
    if FLAGS.eps_use :
        eps = FLAGS.eps_start - (frames / eps_steps) * (FLAGS.eps_start - eps_min)
        return eps if eps > eps_min else eps_min
    else : return eps_min



def evaluate():
    """Evaluated current agent, and records a video with it's performance"""
    envwrap = GymWrapperFactory.make(FLAGS.env,
                                     actrep=FLAGS.action_repeat,
                                     memlen=FLAGS.memory_len,
                                     w=FLAGS.width,
                                     h=FLAGS.height)
    with tf.Session() as sess:
        agent = QlearningAgent(session=sess,
                               action_size=envwrap.action_size,
                               h=FLAGS.height,
                               w=FLAGS.width,
                               channels=FLAGS.memory_len,
                               opt=tf.train.AdamOptimizer(FLAGS.lr))
        sess.run(tf.initialize_all_variables())
        if not os.path.exists(FLAGS.logdir):
            print('ERROR! No', FLAGS.logdir, 'folder found!')
            return
        ckpt = tf.train.latest_checkpoint(FLAGS.logdir)
        if ckpt is not None:
            tf.train.Saver().restore(sess, ckpt)
            agent.update_target()
            print('Session was restored from %s' % ckpt)
        else:
            print('ERROR! No checkpoint found at', FLAGS.logdir)
            return
        envwrap.env.monitor.start(os.path.join(FLAGS.evaldir, FLAGS.env))
        total_reward = 0
        for _ in range(FLAGS.eval_iter):
            s = envwrap.reset()
            terminal = False
            while not terminal:
                reward_per_action = agent.predict_rewards(s)
                s, r, terminal, info = envwrap.step(np.argmax(reward_per_action), test=True)
                total_reward += r
                envwrap.render()
        envwrap.env.monitor.close()
        print('Evaluation finished.')
        print('Average reward per episode: %0.4f' % (total_reward / FLAGS.eval_iter))


def test(agent, env, episodes, while_training = True):
    """Tests agent's performance on given number of games
    :param agent: agent to test
    :param env: environment
    :param episodes: (int) number of testing episodes
    :type agent: agent.QlearningAgent
    :type env: environment.GymWrapper"""
    ep_rewards = []
    ep_q = []
    ep_u = []
    q_loss, u_loss, r_loss = 0., 0., 0.
    for _ in range(episodes):
        rewards_4r, states, actions = [], [], []
        ep_reward = 0
        s = env.reset()
        terminal = False
        len_step = 0
        while not terminal:
            if while_training:
                agent.frame_increment()
            len_step += 1
            states.append(s)
            q_values, u_values = agent.predict_rewards_n_uncertainty(s)
            action = np.argmax(q_values)
            actions.append(action)
            s, r, terminal, info = env.step(action, test=True)
            ep_q.append(np.max(q_values))
            ep_u.append(u_values[:,action])
            rewards_4r.append(r)
            ep_reward += r
        rewards_4q = discount(rewards_4r, 0.99)
        if while_training :
            if FLAGS.batch_avg_reward:
                episodic_rewards_4r = episodic_average(rewards_4r)
            q_l, u_l, r_l = agent.get_loss_values_while_training(np.vstack(states), actions, rewards_4q, episodic_rewards_4r)
        else :
            q_l, u_l, r_l = agent.get_loss_values(np.vstack(states), actions, rewards_4q, rewards_4r)
        q_loss += q_l
        u_loss += u_l
        r_loss += r_l
        ep_rewards.append(ep_reward)
    return ep_rewards, ep_q, ep_u, q_loss/episodes, u_loss/episodes, r_loss/episodes

def episodic_average(x) :
    new_list = []
    for i in range(len(x)) :
        new_list.append(sum(x[i:])/float(len(x)-(i)))
    return new_list

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def train_async_dqn(agent, env, sess, agent_summary, saver, thread_idx=0):
    """Starts Asynchronous one/n-step Q-Learning.
    Can be used as a worker for threading.Thread
    :param agent: learning (asynchronous) agent
    :param env: environment
    :param sess: tensorflow session
    :param agent_summary: object used for summary tensorboard logging
    :param saver: tensorflow session saver
    :param thread_idx: (int) thread index. Thread with index=0 used for target network update and logging
    :type agent: agent.QlearningAgent
    :type env: environment.GymWrapper
    :type sess: tensorflow.Session
    :type agent_summary: agent.AgentSummary
    :type saver: tensorflow.train.Saver"""

    def discount(x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def episodic_average(x) :
        new_list = []
        for i in range(len(x)) :
            new_list.append(sum(x[i:])/float(len(x)-(i)))
        return new_list

    global global_epsilons
    eps_min = random.choice(EPS_MIN_SAMPLES)
    epsilon = update_epsilon(agent.frame, FLAGS.eps_steps, eps_min)
    print('Thread: %d. Sampled min epsilon: %f' % (thread_idx, eps_min))
    last_logging = agent.frame
    last_target_update = agent.frame
    terminal = True
    # Training loop:
    while agent.frame < FLAGS.total_frames:
        batch_states, batch_rewards, batch_actions = [], [], []
        len_step = 0
        if terminal:
            terminal = False
            screen = env.reset_random()
        # Batch update loop:
        while not terminal and len(batch_states) < FLAGS.tmax:
            len_step += 1
            # Increment shared frame counter
            agent.frame_increment()
            batch_states.append(screen)
            # Exploration vs Exploitation, E-greedy action choose
            if random.random() < epsilon:
                action_index = random.randrange(agent.action_size)
            else:
                q_values, u_values = agent.predict_rewards_n_uncertainty(screen)
                # Choose an action index with maximum expected reward
                action_index = np.argmax(q_values + u_values)
            # Execute an action and receive new state, reward for action
            screen, reward, terminal, info_ = env.step(action_index)
            # print(info_)
            reward = np.clip(reward, -1, 1)
            # n-step Q-Learning
            batch_rewards.append(reward)
            batch_actions.append(action_index)
        # last reward to be discounted value for q updates
        if FLAGS.batch_avg_reward:
            ep_batch_avg_rewards_ = episodic_average(batch_rewards)
        if len(batch_states) == (FLAGS.tmax - 1) and not terminal:
            batch_rewards[-1] += FLAGS.gamma * agent.predict_target(screen)
        # Accumulate Rewards for n-step
        rewards_plus = np.asarray(batch_rewards)
        batch_discounted_rewards4Q = discount(rewards_plus, FLAGS.gamma)
        # Apply asynchronous gradient update to shared agent
        if FLAGS.batch_avg_reward :
            agent.train(np.vstack(batch_states), batch_actions, batch_discounted_rewards4Q, ep_batch_avg_rewards_)
        else :
            agent.train(np.vstack(batch_states), batch_actions, batch_discounted_rewards4Q, batch_rewards)
        # Anneal epsilon
        epsilon = update_epsilon(agent.frame, FLAGS.eps_steps, eps_min)
        global_epsilons[thread_idx] = epsilon  # Logging
        # Logging and target network update
        if thread_idx == 0:
            if agent.frame - last_target_update >= FLAGS.update_interval:
                last_target_update = agent.frame
                agent.update_target()
            if agent.frame - last_logging >= FLAGS.log_interval and terminal:
                last_logging = agent.frame
                saver.save(sess, os.path.join(FLAGS.logdir, "sess.ckpt"), global_step=agent.frame)
                print('Session saved to %s' % FLAGS.logdir)
                if FLAGS.env != "Solaris-v0":
                    episode_rewards, episode_q, episode_u, q_loss, u_loss, r_loss = test(agent, env,
                                                                                         episodes=FLAGS.test_iter,
                                                                                         while_training=True)
                else :
                    print('solaris testing....')
                    episode_rewards, episode_q, episode_u, q_loss, u_loss, r_loss = test(agent, env,
                                                                                         episodes=15,
                                                                                         while_training=True)
                avg_r = np.mean(episode_rewards)
                avg_q = np.mean(episode_q)
                avg_u = np.mean(episode_u)
                avg_eps = np.mean(global_epsilons)
                print("%s. Avg.Ep.R: %.4f. Avg.Ep.Q: %.2f. Avg.Ep.U: %.2f. Avg.Eps: %.2f. T: %d" %
                      (str(datetime.now())[11:19], avg_r, avg_q, avg_u, avg_eps, agent.frame))
                agent_summary.write_summary({
                    'total_frame_step': agent.frame,
                    'episode_avg_reward': avg_r,
                    'avg_q_value': avg_q,
                    'avg_u_value': avg_u,
                    'avg_q_loss': q_loss,
                    'avg_u_loss': u_loss,
                    'avg_r_loss': r_loss,
                    'epsilon': avg_eps
                })
    global training_finished
    training_finished = True
    print('Thread %d. Training finished. Total frames: %s' % (thread_idx, agent.frame))


def run(worker):
    """Launches worker asynchronously in 'FLAGS.threads' threads
    :param worker: worker function"""
    print('Starting %s threads.' % FLAGS.threads)
    processes = []
    envs = []
    for _ in range(FLAGS.threads):
        env = GymWrapperFactory.make(FLAGS.env,
                                     actrep=FLAGS.action_repeat,
                                     memlen=FLAGS.memory_len,
                                     w=FLAGS.width,
                                     h=FLAGS.height)
        envs.append(env)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        agent = QlearningAgent(session=sess,
                               action_size=envs[0].action_size,
                               h=FLAGS.height,
                               w=FLAGS.width,
                               channels=FLAGS.memory_len,
                               opt=tf.train.AdamOptimizer(FLAGS.lr),
                               gamma = FLAGS.gamma)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        sess.run(tf.global_variables_initializer())
        if not os.path.exists(FLAGS.logdir):
            os.makedirs(FLAGS.logdir)
        ckpt = tf.train.latest_checkpoint(FLAGS.logdir)
        if ckpt is not None:
            saver.restore(sess, ckpt)
            agent.update_target()
            print('Restoring session from %s' % ckpt)
        summary = AgentSummary(FLAGS.logdir, agent, FLAGS.env)
        for i in range(FLAGS.threads):
            processes.append(th.Thread(target=worker, args=(agent, envs[i], sess, summary, saver, i,)))
        for p in processes:
            p.daemon = True
            p.start()
        while not training_finished:
            if FLAGS.render:
                for i in range(FLAGS.threads):
                    envs[i].render()
            time.sleep(.01)
        for p in processes:
            p.join()


def main():
    if FLAGS.eval:
        evaluate()
    else:
        run(train_async_dqn)


if __name__ == '__main__':
    main()
