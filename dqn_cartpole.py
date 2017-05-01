from __future__ import division
from __future__ import print_function
import argparse

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

ENV_NAME = 'CartPole-v0'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', 
                        choices=('train', 'test'))
    parser.add_argument('--weights-filename', type=str, default=None)
    parser.add_argument('--agent-memory-limit', type=int, default=50000)
    parser.add_argument('--agent-warmup-steps', type=int, default=10)
    parser.add_argument('--target-model-update', type=int, default=0.01)
    parser.add_argument('--disable-double-dqn', action='store_false')
    parser.add_argument('--enable-dueling-network', action='store_true')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--training-steps', type=int, default=50000)
    parser.add_argument('--test-episodes', type=int, default=10)
    parser.add_argument('--render-test', action='store_true')
    args = parser.parse_args()

    # Create the environment
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)
    # Extract the number of actions from environment
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    # Build DQN agent
    memory = SequentialMemory(limit=args.agent_memory_limit, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=args.agent_warmup_steps,
        target_model_update=args.target_model_update,
        policy=policy,
        enable_double_dqn=args.disable_double_dqn,
        enable_dueling_network=args.enable_dueling_network)
    dqn.compile(Adam(lr=args.lr), metrics=['mae'])

    weights_filename = 'dqn_{}_weights.h5f'.format(ENV_NAME)
    if args.weights_filename is not None:
        weights_filename = args.weights_filename

    if args.mode == 'train':
        checkpoint_weights_filename = 'dqn_' + ENV_NAME + '_weights_{step}.h5f'
        log_filename = 'dqn_{}_log.json'.format(ENV_NAME)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=10000)]
        callbacks += [FileLogger(log_filename, interval=100)]
        
        dqn.fit(env, nb_steps=args.training_steps, callbacks=callbacks)

        dqn.save_weights(weights_filename, overwrite=True)

        dqn.test(env, nb_episodes=args.test_episodes, visualize=args.render_test)
    elif args.mode == 'test':
        dqn.load_weights(args.weights_filename)
        dqn.test(env, nb_episodes=args.test_episodes, visualize=args.render_test)

if __name__ == '__main__':
    main()
