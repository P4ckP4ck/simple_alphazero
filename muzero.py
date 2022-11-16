import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import config as cfg
import helper as hlp
from networks import alphazero_net as Network
from node import Node
from top_est_env_alpha import TopologyEstimatorEnvironment as Env


# Stops tensorflow from allocating all GPU memory, to enable multithreading with several networks
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def predict(root: Node) -> [np.array, float]:
    """
    Wrapper function to perform the prediction on a root state
    :param root: root node to predict priors and values from
    :return: priors and values of the root-node
    """
    child_priors, value = root._network.predict(root._state)
    updated_priors = [np.exp(child_priors[0][a]) for a in cfg.action_range]
    priors_sum = sum(updated_priors)
    priors = [updated_priors[a] ** (1 / cfg.prior_temp) / priors_sum ** (1 / cfg.prior_temp) for a in cfg.action_range]
    return priors, value


def run_episode(network: Network,
                scaler: hlp.Scaler,
                replay_buffer: hlp.PrioritizedExperienceReplay,
                environment: Env) -> float:
    """
    Main simulation loop for the algorithm
    :param network: Neural network to perform predictions with
    :param scaler: Scaler object
    :param replay_buffer: PrioritizedExperienceReplay object
    :param environment: Environment object
    :return: Accumulated reward
    """
    # Initializing simulation
    prog_bar = tqdm(range(cfg.max_moves))
    scaler.refresh()
    state: np.array = environment.reset()
    total_reward: float = 0.

    # Main simulation loop
    for _ in prog_bar:
        root = Node(network=network,
                    environment=environment,
                    scaler=scaler,
                    state=state)

        # Perform Monte Carlo Tree Search
        for _ in range(cfg.num_simulations):
            root.explore()

        # Act according to Tree Search finding and store results
        action = root.get_action()
        replay_buffer.append(state=state,
                             priors=root.get_priors(),
                             value=root.value)
        state, reward, done = environment.step(action)
        environment.plot_history()
        total_reward += reward
        prog_bar.set_description(f"Total Reward: {total_reward:.2f}")
        if done:
            prog_bar.set_description(f"Total Reward: {total_reward:.2f}")
            prog_bar.close()
            break
    return total_reward


def plot_stats(reward_hist: list) -> None:
    """
    Saving statistics as image to file
    :param reward_hist: List containing the training history
    """
    plt.hist(reward_hist)
    plt.savefig("reward_histogram.jpg")
    plt.close()
    plt.plot(reward_hist)
    plt.plot(pd.Series(reward_hist).rolling(50, min_periods=1).mean())
    plt.legend(['Rewards per Episode', '50 mean rewards'], loc='upper left')
    plt.savefig("reward_history.jpg")
    plt.close()


if __name__ == "__main__":
    # Enabling PyCharm to close matplotlib windows during runtime
    plt.ioff()

    # Initializing helper objects
    scaler = hlp.Scaler()
    net = Network()
    replay_buffer = hlp.PrioritizedExperienceReplay(net)
    total_rewards = []
    net.train_network(replay_buffer)
    environment = Env()

    # Starting training cycles
    for _ in range(cfg.complete_cycles):
        for _ in range(cfg.num_episodes):
            ep_reward = run_episode(net, scaler, replay_buffer, environment)
            total_rewards.append(ep_reward)

        # Saving simulations and train the networks
        print(f"Mean total reward: {np.mean(total_rewards):.2f}")
        replay_buffer.write_to_disc()
        replay_buffer.reload_buffer()
        net.train_network(replay_buffer)
        plot_stats(total_rewards)
        cfg.explore_fraction *= cfg.explore_decay
        cfg.explore_fraction = max(cfg.explore_fraction, cfg.min_explore)
