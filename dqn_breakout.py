from __future__ import division
from __future__ import print_function
import argparse

import numpy as np
import gym
from PIL import Image
# from wrappers.frameBufferWrapper import FrameBufferWrapper

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Permute
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

IMAGE_SIZE = (84, 84)

class BreakoutProcessor(Processor):
    def process_reward(self, reward):
        return np.clip(reward, -1.0, 1.0)

    def process_observation(self, observation):
        # Observation image is in color should be changed to black and white
        img = Image.fromarray(observation) # change numpy array to image object
        img = img.resize(IMAGE_SIZE) # resize img to 84 x 84
        img = img.convert('L') # img.save('image.png') # convert image to black and white
        img.save('image.png')
        processed_observation = np.array(img) # change img obejct back to numpy with height and width 84 x 84
        # print('image shape: {}'.format(processed_observation.shape))
        assert processed_observation.shape == IMAGE_SIZE
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.0
        return processed_batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Breakout-v0')
    parser.add_argument('--mode', type=str, default='train',
                        choices=('test', 'train', 'resume'))
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--agent-memory-limit', type=int, default=1000000)
    parser.add_argument('--policy-value-max', type=float, default=1.0)
    parser.add_argument('--policy-value-min', type=float, default=0.1)
    parser.add_argument('--policy-value-test', type=float, default=0.05)
    parser.add_argument('--policy-nb-steps', type=int, default=1500000)
    parser.add_argument('--agent-warmup-steps', type=int, default=50000)
    parser.add_argument('--agent-gamma', type=float, default=0.99)
    parser.add_argument('--agent-train-interval', type=int, default=4)
    parser.add_argument('--agent-delta-clip', type=float, default=1.0)
    parser.add_argument('--target-update-model', type=int, default=10000)
    parser.add_argument('--disable-double-dqn', action='store_false')
    parser.add_argument('--enable-dueling-network', action='store_true')
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--training-steps', type=int, default=50000000)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--model-checkpoint-interval', type=int, default=200000)
    parser.add_argument('--file-logger-interval', type=int, default=100)
    parser.add_argument('--log-interval', type=int, default=10000)
    parser.add_argument('--test-episodes', type=int, default=10)
    parser.add_argument('--render-test', action='store_true')
    args = parser.parse_args()

    # Create gym environment
    env = gym.make(args.env)
    np.random.seed(args.seed)
    env.seed(args.seed)
    # Extract the number of actions from environment
    nb_actions = env.action_space.n
    # env = FrameBufferWrapper(env, 4)
    input_shape = (4,) + IMAGE_SIZE

    # Model based on Mnih et. al (2013) and Mnih et. al. (2015)
    # Model consists of 3 CNN and 2 linear neural networks
    model = Sequential()
    # Permute Layer used to get correct input shape (None, 84, 84, 4) to convolution layer
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(32, (8, 8), strides=4))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (4, 4), strides=2))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides=1))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    breakout_processor = BreakoutProcessor()
    memory = SequentialMemory(limit=args.agent_memory_limit, window_length=4)
    policy = LinearAnnealedPolicy(
        inner_policy=EpsGreedyQPolicy(),
        attr='eps',
        value_max=args.policy_value_max,
        value_min=args.policy_value_min,
        value_test=args.policy_value_test,
        nb_steps=args.policy_nb_steps)

    dqn = DQNAgent(
        model=model,
        processor=breakout_processor,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=args.agent_warmup_steps,
        target_model_update=args.target_update_model,
        policy=policy,
        train_interval=args.agent_train_interval,
        delta_clip=args.agent_delta_clip,
        gamma=args.agent_gamma,
        enable_double_dqn=args.disable_double_dqn,
        enable_dueling_network=args.enable_dueling_network)

    dqn.compile(Adam(lr=args.lr), metrics=['mae'])

    weights_filename = 'dqn_{}_weights.h5f'.format(args.env)
    if args.weights is not None:
        weights_filename = args.weights

    if args.mode == 'resume':
        # if resuming after 1000000 steps of training,
        # make sure not to linearly anneal epsilon
        dqn.load_weights(weights_filename)
        args.mode = 'train'

    if args.mode == 'train':
        checkpoint_weights_filename = 'dqn_' + args.env + '_weights_{step}.h5f'
        log_filename = 'dqn_{}_log.json'.format(args.env)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=args.model_checkpoint_interval)]
        callbacks += [FileLogger(log_filename, interval=args.file_logger_interval)]
        dqn.fit(env, nb_steps=args.training_steps, callbacks=callbacks, log_interval=args.log_interval)
        dqn.save_weights(weights_filename, overwrite=True)
        dqn.test(env, nb_episodes=args.test_episodes, visualize=args.render_test)
    else:
        dqn.load_weights(weights_filename)
        dqn.test(env, nb_episodes=args.test_episodes, visualize=args.render_test)


if __name__ == '__main__':
    main()
