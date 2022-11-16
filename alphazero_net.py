"""
P. Lehnen - Masters Thesis - Neural networks for topology estimation of power grids - 30.3.2022
Experiment 7: AlphaZero Network
v0.0.1
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

import config as cfg


class Network:
    def __init__(self):
        self._alpha_net = self._alpha_network()
        self._load_weights()

    @staticmethod
    def _alpha_network() -> Model:
        """
        Defining the Keras model
        :return: Compiled neural network model
        """
        inp = kl.Input(shape=cfg.state_shape)
        flat = kl.Dense(64,
                        activation="leaky_relu")(inp)
        flat = kl.Dense(64,
                        activation="leaky_relu")(flat)
        flat = kl.Dense(64,
                        activation="leaky_relu")(flat)

        x = kl.Dense(32,
                     activation="leaky_relu")(flat)

        out_prior = kl.Dense(cfg.action_len, activation="softmax", name="prior")(x)
        out_value = kl.Dense(cfg.bins, activation="softmax", name="value")(x)

        rep_net = Model(inp, [out_prior, out_value])
        rep_net.compile(optimizer=tf.keras.optimizers.Adam(cfg.learning_rate),
                        loss=(tf.losses.categorical_crossentropy,
                              tf.losses.categorical_crossentropy))
        return rep_net

    def _get_single_inference(self, state: np.array) -> (float, float, np.array, np.array):
        """
        Function for a single state prediction
        :param state: A single state
        :return: Predictions
        """
        child_priors, value = self._alpha_net.predict(np.expand_dims(state, axis=0),
                                                      verbose=0)
        return child_priors, self.distribution_to_scalar(value)

    def _get_inference(self, state: np.array) -> (float, float, np.array, np.array):
        """
        Function for a batch of state predictions
        :param state: batch of states
        :return: Predictions for each state in batch
        """
        child_priors, value = self._alpha_net.predict(state)
        return child_priors, self.distribution_to_scalar(value)

    def predict(self, state: np.array) -> np.array:
        """
        Wrapper to make predictions easier. Checks dimensions and chooses the right prediction function
        :param state: single state OR batch of states
        :return: Predictions of state(s)
        """
        if len(state.shape) == 2:
            return self._get_inference(state)
        if len(state.shape) == 1:
            return self._get_single_inference(state)
        else:
            raise AssertionError

    def train_network(self, replay_buffer: "ReplayBuffer") -> None:
        """
        Wrapper for simpler training of the neural network
        :param replay_buffer: Replay buffer object to draw samples from
        """
        print("Training network...")
        for _ in range(cfg.training_loops):
            states, targets = replay_buffer.get_samples()
            loss_hist = self._alpha_net.fit(x=states,
                                            y=targets,
                                            epochs=cfg.train_epochs,
                                            batch_size=cfg.batch_size,
                                            verbose=0)
        self._save_networks()
        print(f"Loss: {loss_hist.history['loss'][-1]:.5f}")

    @staticmethod
    def _scale_gradient(tensor: tf.Tensor, scale):
        """
        Scaling the gradient during backpropagation
        :param tensor: gradient tensor
        :param scale: scaling factor
        :return: scaled tensor
        """
        return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)

    def _save_networks(self):
        """
        Handler for saving the network
        """
        self._alpha_net.save_weights("save/alpha_net.h5")

    def _load_weights(self):
        """
        Handler for loading the network including error checking
        """
        try:
            self._alpha_net.load_weights("save/alpha_net.h5")
        except:
            print("Network weights not found. Initializing...")

    @staticmethod
    def reward_transform(x: np.ndarray, var_eps: float = 0.001) -> np.ndarray:
        """
        Transforms the given reward to be more continous over the expected range. Reduces outliers
        :param x: Reward to transform
        :param var_eps: Smoothing value
        :return: Transformed reward
        """
        return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + var_eps * x

    @staticmethod
    def inverse_reward_transform(x: np.ndarray, var_eps: float = 0.001) -> np.ndarray:
        """
        Inverses the reward-transformation
        :param x: Transformed reward to inverse
        :param var_eps: Smoothing value
        :return: Inversed reward
        """
        return np.sign(x) * (((np.sqrt(1 + 4 * var_eps * (np.abs(x) + 1 + var_eps)) - 1) / (2 * var_eps)) ** 2 - 1)

    def distribution_to_scalar(self, distribution: np.ndarray) -> np.ndarray:
        """
        Transforms a distribution back to a scalar
        :param distribution: The distribution to transform
        :return: scalar value
        """
        bins = np.arange(-cfg.support_size, cfg.support_size + 1)
        y = np.dot(distribution, bins)
        value = self.inverse_reward_transform(y)
        return value

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
