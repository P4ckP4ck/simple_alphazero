import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

import config as cfg
import helper as hlp
from alphazero_net import Network
from node import Node
import gym

import matplotlib
matplotlib.use("TkAgg")


# Stops tensorflow from allocating all GPU memory, to enable multithreading with several networks
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def run_episode(network: Network,
                scaler: hlp.Scaler,
                environment,
                replay_buffer) -> float:
    """
    Main simulation loop for the algorithm
    :param network: Neural network to perform predictions with
    :param scaler: Scaler object
    :param environment: Environment object
    :return: Accumulated reward
    """
    # Initializing simulation

    scaler.refresh()
    state: np.array = environment.reset()
    total_reward: float = 0.

    # Main simulation loop
    for step in range(cfg.max_moves):
        root = Node(network, environment, scaler, state)

        # Perform Monte Carlo Tree Search
        for _ in range(cfg.num_simulations):
            root.explore()

        # Act according to Tree Search finding and store results
        action = root.get_action()
        next_state, reward, done, info = environment.step(action)
        # environment.render()
        if not step == 0:
            val = old_reward + root.value * cfg.discount if not done else reward
            replay_buffer.append(state=old_state,
                                 priors=old_priors,
                                 value=val)
        old_state, old_priors, old_reward = next_state, root.get_priors(), reward
        total_reward += reward
        if done:
            break
    return total_reward


def get_target(node):
    while node.visit_count != 1:
        visits = []
        for child in node._children.values():
            visits.append(child.visit_count)
        node = node._children[np.argmax(visits)]
    return node.value

if __name__ == "__main__":
    # Initializing helper objects
    plt.ioff()
    environment = gym.make('CartPole-v1')#, render_mode="rgb_array")
    scaler = hlp.Scaler()
    network = Network()
    replay_buffer = hlp.PrioritizedExperienceReplay(network)
    network.train_network(replay_buffer)
    total_rewards = []
    # Starting simulation cycles
    for _ in range(cfg.complete_cycles):

        prog_bar = tqdm(range(cfg.num_episodes))
        for _ in prog_bar:
            ep_reward = run_episode(network, scaler, environment, replay_buffer)
            total_rewards.append(ep_reward)
            prog_bar.set_description(f"Last / Total Reward: {ep_reward:.2f} / {np.mean(total_rewards):.2f}")
        prog_bar.close()

        replay_buffer.write_to_disc()
        replay_buffer.reload_buffer()
        network.train_network(replay_buffer)

        cfg.explore_fraction *= cfg.explore_decay
        cfg.explore_fraction = max(cfg.explore_fraction, cfg.min_explore)

        print(f"Mean total reward: {np.mean(total_rewards):.2f}\n"
              f"Epsilon          : {cfg.explore_fraction}")

        pd.Series(total_rewards).plot().get_figure()
        plt.plot(pd.Series(total_rewards).rolling(250, min_periods=1).mean())
        plt.savefig("history_mean.jpg")
        plt.close()
