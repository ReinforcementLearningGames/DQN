from __future__ import division
from __future__ import print_function
import argparse

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from rl.callbacks import ModelIntervalCheckpoint

ENV = 'CartPole-v0'

def main():
    # use parser to get command arguments to make it easier to change parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--memory_limit', type=int, default=40000)
    parser.add_argument('--disable-ddqn', action='store_false')
    parser.add_argument('--target-model-update', type=int, default=0.01)
    parser.add_argument('--discount-factor', type=float, default=0.99)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--training-steps', type=int, default=40000)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--test-episodes', type=int, default=10)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    # create new environment, seed environment, and get actions from environment
    env = gym.make(ENV)
    if args.seed is not None:
        env.seed(args.seed)
    nb_actions = env.action_space.n

    # build model
    model = Sequential()
    model.add(Flatten(input_shape=(1, ) + env.observation_space.shape))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    # make experience memory of size
    memory = SequentialMemory(limit=args.memory_limit, window_length=1)
    # pick a simple policy for cartpole
    policy = BoltzmannQPolicy()

    # create the DQN agent
    dqn = DQNAgent(
        model=model,
        policy=policy,
        enable_double_dqn=args.disable_ddqn,
        memory=memory,
        target_model_update=args.target_model_update,
        nb_steps_warmup=args.warmup,
        gamma=args.discount_factor,
        nb_actions=nb_actions
    )
    dqn.compile(Adam(lr=args.lr), metrics=['mae'])

    weights_filename = 'cartpole_weights.h5f'

    if args.mode == 'train':
        # make filenames for saving weights at checkpoints
        checkpoint_weights_filename = 'cartpole_weights_{step}.h5f'
        # add some callback to see reward and loss
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=10000)]
        # fit model for 40k steps
        dqn.fit(env, nb_steps=args.training_steps, callbacks=callbacks)
        # Save final weights
        dqn.save_weights(weights_filename, overwrite=True)
    elif args.mode == 'test':
        # Check if custom weights file. If so, assign new weights filename
        if args.weights is not None:
            weights_filename = args.weights
        # Load weights from file
        dqn.load_weights(weights_filename)
        # Test model
        dqn.test(env, nb_episodes=args.test_episodes, visualize=args.render)


if __name__ == '__main__':
    main()
