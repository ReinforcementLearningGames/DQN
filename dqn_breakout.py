from __future__ import division
from __future__ import print_function
import argparse

from PIL import Image

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Permute
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

from rl.core import Processor
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


OBSERVATION_SHAPE = (84, 84)
ENV = 'Breakout-v0'

class BreakoutProcessor(Processor):
    def process_observation(self, observation):
        # convert from numpy array to image object
        img = Image.fromarray(observation)
        # resize and convert to grayscale
        img = img.resize(OBSERVATION_SHAPE).convert('L')
        #  convert back to numpy array
        processed_observation = np.array(img)
        # make sure observation is correct shape
        assert processed_observation.shape == OBSERVATION_SHAPE
        # convert to integer to save storage in experience memory
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        # change back to float and scale between 0.0 and 1.0
        return batch.astype('float32') / 255.0

    def process_reward(self, reward):
        # clip rewards to between -1 and 1
        # anything less than 0 becomes -1
        # anything greater than 0 becomes 1
        return np.clip(reward, -1.0, 1.0)

def main():
    # use parser to get command arguments to make it easier to change parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--memory-limit', type=int, default=1000000)
    parser.add_argument('--frames', type=int, default=4)
    parser.add_argument('--initial-exploration', type=float, default=1.0)
    parser.add_argument('--final-exploration', type=float, default=0.1)
    parser.add_argument('--final-exploration-frame', type=int, default=1000000)
    parser.add_argument('--disable-ddqn', action='store_false')
    parser.add_argument('--train-interval', type=int, default=4)
    parser.add_argument('--target-model-update', type=int, default=10000)
    parser.add_argument('--discount-factor', type=float, default=0.99)
    parser.add_argument('--warmup', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--training-steps', type=int, default=50000000)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--test-episodes', type=int, default=10)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    # create new environment, seed environment, and get actions from environment
    env = gym.make(ENV)
    if args.seed is not None:
        env.seed(args.seed)
    nb_actions = env.action_space.n

    input_shape = (args.frames, ) + OBSERVATION_SHAPE

    # build model based on what is desribed in Mnih et al (2015)
    model = Sequential()
    # input shape = (84, 84, 4)
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    # make experience memory of size 1000000, stack 4 frames into one
    memory = SequentialMemory(limit=args.memory_limit, window_length=args.frames)
    # create processor object
    processor = BreakoutProcessor()
    # linearly anneal epsilon greedy policy over 1000000 steps
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=args.initial_exploration,
        value_min=args.final_exploration,
        value_test=0.05,
        nb_steps=args.final_exploration_frame
    )

    # create the DQN agent with parameters from Mnih et al (2015)
    dqn = DQNAgent(
        model=model,
        policy=policy,
        enable_double_dqn=args.disable_ddqn,
        memory=memory,
        processor=processor,
        train_interval=args.train_interval,
        target_model_update=args.target_model_update,
        gamma=args.discount_factor,
        nb_steps_warmup=args.warmup,
        nb_actions=nb_actions,
        delta_clip=1.0
    )

    # compile agent with Adam instead of RMSProp
    dqn.compile(Adam(lr=args.lr), metrics=['mae'])

    # default weights filename
    weights_filename = 'breakout_weights.h5f'

    if args.mode == 'train':
        # make filenames for saving weights at checkpoints
        checkpoint_weights_filename = 'breakout_weights_{step}.h5f'
        # add some callback to see reward and loss
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
        # fit model for 50M steps
        dqn.fit(env, nb_steps=args.training_steps, callbacks=callbacks)
        # Save final weights
        dqn.save_weights(weights_filename, overwrite=True)
    elif args.mode == 'test':
        # Check for custom weights file. If so, assign new weights filename
        if args.weights is not None:
            weights_filename = args.weights
        # Load weights from file
        dqn.load_weights(weights_filename)
        # Test model
        dqn.test(env, nb_episodes=args.test_episodes, visualize=args.render)

if __name__ == '__main__':
    main()
