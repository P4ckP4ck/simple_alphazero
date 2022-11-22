import os
import random
import time
from typing import Dict, List

import numpy as np

import config as cfg
from alphazero_net import Network



class Scaler:
    # Class for scaling the values during tree traversal
    def __init__(self):
        self.weights: Dict[str, float] = {"max": 1e-10, "min": -1e-10}

    def update(self, value: float) -> None:
        """
        Checks if the value is a new minimum or maximum value and updates accordingly
        :param value: Observed value
        """
        self.weights["max"] = max(self.weights["max"], value)
        self.weights["min"] = min(self.weights["min"], value)

    def normalize(self, value: float) -> float:
        """
        Normalizing the observed value
        :param value: Observed value
        :return: Normalized value according to observed min/max values
        """
        return (value - self.weights["min"]) / (self.weights["max"] - self.weights["min"])

    def refresh(self):
        """
        Resets the scaler after each simulation
        """
        self.weights: Dict[str, float] = {"max": 1e-10, "min": -1e-10}


class Episode:
    # Class for storing relevant informations of the current episode
    def __init__(self):
        self.environment = None#Env()
        self.state = self.environment.reset()
        self.done = False
        self.states = []
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []

    @property
    def total_reward(self) -> float:
        """
        Property for the achieved total reward
        :return: Sum of all collected rewards
        """
        return sum(self.rewards)

    @property
    def is_terminal(self) -> bool:
        """
        Checks if a terminal state is reached
        :return: Boolean for terminal state
        """
        return self.done

    def get_state(self, idx: int) -> np.array:
        """
        Collects the specified state
        :param idx: Index of returned state
        :return: State at specified index
        """
        return self.states[idx]

    @property
    def length(self) -> int:
        """
        Checks the length of stored states
        :return: Count of states this episode
        """
        return len(self.states)

    def apply(self, action: int) -> None:
        """
        Takes an action and traverses the environment accordingly. Saves observations
        :param action: Chosen action index
        """
        self.state, reward, self.done, _ = self.environment.step(action)
        self.states.append(self.state)
        self.rewards.append(reward)
        self.history.append(action)

    def store_search_statistics(self, root) -> None:
        """
        Stores the statistics of the simulation for later use in the training phase for the priors
        :param root: Node which was searched
        """
        visit_count_dict = root.get_visit_count()
        visit_count = np.array([visit_count_dict[a] for a in visit_count_dict])
        sum_visits = sum(visit_count)
        self.child_visits.append(visit_count / sum_visits)
        self.root_values.append(root.mean_value)

    def get_priors(self) -> np.array:
        """
        Returns the last probability distribution of node visits
        :return: priors of last stored node
        """
        return self.child_visits[-1]

    def get_value(self) -> float:
        """
        Return the value of the last stored node
        :return: Value of last node
        """
        return self.root_values[-1]

    def get_target(self, state_index: int, num_unroll_steps: int, td_steps: int):
        """
        Calculate the training targets
        :param state_index: State index to train upon
        :param num_unroll_steps: Unroll steps in the future to calculate target with (k in paper)
        :param td_steps: temporal difference / lambda
        :return: calculated target values
        """
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * cfg.discount**td_steps
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * cfg.discount**i

            if 0 < current_index <= len(self.rewards):
                last_reward = self.rewards[current_index - 1]
            else:
                last_reward = 0

            if current_index < len(self.root_values):
                targets.append((value, last_reward, self.child_visits[current_index]))
        return targets


class PrioritizedExperienceReplay:
    # This class samples training data with higher td-errors more often, resulting in faster training
    def __init__(self, network: Network, main_thread=True):
        self._buffer: List[np.array, np.array, float] = []
        self._network = network
        self._errors_prob: np.array = None
        if main_thread:
            self.reload_buffer()

    @property
    def size(self) -> int:
        """
        Check the current length of the buffer
        :return: Buffer length as integer
        """
        return len(self._buffer)

    def append(self, state, priors, value) -> None:
        """
        Adding the last observations to the buffer and forgetting the first ones, if buffer reached max capacity
        :param state: Observed states
        :param priors: Observed priors
        :param value: Observed values
        """
        if self.size > cfg.buffer_size:
            self._buffer.pop(0)
        self._buffer.append([state, priors, value])

    def get_samples(self) -> (np.array, List):
        """
        Draws a number of samples with more weight to those samples with higher td-error
        :return: Training batch
        """
        self._calc_errors()
        sample = random.choices(population=self._buffer,
                                k=min(len(self._buffer), cfg.training_samples),
                                weights=self._errors_prob)
        states, priors, values = map(np.asarray, zip(*sample))
        return states, [priors, self.scalar_to_distribution(values)]

    def _calc_errors(self) -> None:
        """
        Calculates the td-errors for the current samples in buffer
        """
        states, priors, values = map(np.asarray, zip(*self._buffer))
        target_priors, target_values = self._network.predict(states)
        td_errors = np.squeeze(target_values) - np.squeeze(values) + np.sum(target_priors - priors, axis=1)
        td_errors = (np.abs(td_errors) + 0.01) ** 0.6
        self._errors_prob = td_errors / sum(td_errors)

    def write_to_disc(self) -> None:
        """
        Saving buffer to file
        """
        np.save(f"buffer/buffer_{cfg.THREAD_NAME}.npy", self._buffer)

    def partial_write_to_disc(self) -> None:
        """
        If buffer is not running on the main thread, this will save the current state
        and refresh the buffer to save memory
        """
        np.save(f"buffer/xtemp_buffer_{int(time.time()*10000)}.npy", self._buffer)
        self._buffer = []

    def reload_buffer(self) -> None:
        """
        If training was paused this function loads the buffer from memory.
        If several buffers are active, it will reload temporary buffer savings
        from sub-threads and integrate them to the main buffer
        """
        buffer_files = os.listdir("./buffer")
        try:
            self._buffer = np.load(f"buffer/buffer_{cfg.THREAD_NAME}.npy", allow_pickle=True).tolist()
            for file in buffer_files[1:]:
                self._buffer += np.load(f"buffer/{file}", allow_pickle=True).tolist()
                os.remove(f"buffer/{file}")
            self._buffer = self._buffer[-cfg.buffer_size:]
            self.write_to_disc()
        except:
            print("No ReplayBuffer found. Initializing...")

    @staticmethod
    def reward_transform(x: np.ndarray, var_eps: float = 0.001) -> np.ndarray:
        """
        The buffer needs to be able to transform the rewards
        :param x: reward
        :param var_eps: Scaling factor
        :return: Transformed reward
        """
        return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + var_eps * x

    def scalar_to_distribution(self, x: np.ndarray) -> np.ndarray:
        """
        Transforms a scalar value to a distribution. Splits the value between the supports
        of the distribution automatically
        :param scalar: Scalar value
        :return:  Distribution of the splitted scalar
        """
        transformed = np.clip(self.reward_transform(x.astype(np.float32)), -cfg.support_size, cfg.support_size - 1e-8)
        floored = np.floor(transformed).astype(int)
        prob = transformed - floored
        bins = np.zeros((len(x), 2 * cfg.support_size + 1))
        for idx in range(len(x)):
            bins[idx, floored[idx] + cfg.support_size] = 1 - prob[idx]
            bins[idx, floored[idx] + cfg.support_size + 1] = prob[idx]
        return bins
