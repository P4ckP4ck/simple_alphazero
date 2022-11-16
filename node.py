"""
P. Lehnen - Masters Thesis - Neural networks for topology estimation of power grids - 30.3.2022
Experiment 7: Search Tree Functions. Nodes are building blocks of the search tree
v0.0.1
"""
import math
from typing import Dict, List

import numpy as np

import config as cfg
import helper as hlp
from alphazero_net import Network
from copy import deepcopy


class Node:
    # Main class containing the methods for Monte Carlo Tree Search
    def __init__(self,
                 network: Network,
                 environment,
                 scaler: hlp.Scaler,
                 state: np.array = None,
                 parent: 'Node' = None,
                 prior: float = None,
                 reward: float = 0,
                 done: bool = False,
                 depth: int = 0) -> None:

        # public attributes
        self.visit_count: int = 0
        self.reward = reward
        self.done = done
        self.depth = depth
        self.solved = False

        # private attributes
        self._environment = environment
        self._value_sum: int = 0
        self._children: Dict[int, Node] = {}
        self._network: Network = network
        self._scaler: hlp.Scaler = scaler
        self._state: np.array = state
        self._parent: Node = parent
        self._prior: float = prior

        if self.is_root:
            self.expand()

    @property
    def is_root(self) -> bool:
        """
        Flag if this node is a root node
        :return: Boolean value if root or not
        """
        return self._parent is None

    @property
    def value(self) -> float:
        """
        Calculates the value of this node
        :return: Value
        """
        if self.visit_count == 0:
            return 0
        return self._value_sum / self.visit_count

    @property
    def ucb_score(self) -> float:
        """
        Calculate the ucb score of this node
        :return: UCB-Score
        """
        if self.is_root or self.done or self.depth == cfg.max_search_depth:
            return -np.inf

        pb_c = math.log((self._parent.visit_count + cfg.pb_c_base + 1) /
                        cfg.pb_c_base) + cfg.pb_c_init
        pb_c *= math.sqrt(self._parent.visit_count) / (self.visit_count + 1)

        prior_score = pb_c * self._prior

        if self.visit_count > 0 and self._prior != 0:
            value_score = self.reward + cfg.discount * self._scaler.normalize(self.value)
        else:
            value_score = 0

        return prior_score + value_score

    def rollout(self, value_sn: float) -> None:
        """
        Perform rollout, propagating the found values up the trees
        :param value_sn:
        """
        self._value_sum += value_sn
        self.visit_count += 1
        value_sn = self.reward + cfg.discount * value_sn
        if not self.is_root:
            self._scaler.update(value_sn)
            self._parent.rollout(value_sn)

    def predict(self) -> (List, float):
        """
        Predict follow-up priors and value from this node
        :return: priors and value of next states
        """
        child_priors, value = self._network.predict(self._state)
        updated_priors = [np.exp(child_priors[0][a]) for a in cfg.action_range]
        priors_sum = sum(updated_priors)
        priors = [updated_priors[a] ** (1 / cfg.prior_temp) / priors_sum ** (1 / cfg.prior_temp)
                  for a in cfg.action_range]
        return priors, value

    def expand(self) -> None:
        """
        Expansion phase of the MCTS-algorithm, opening up the tree and collecting priors for follow-up states
        """
        priors, value = self.predict()
        self._value_sum += value
        for p, a in zip(priors, cfg.action_range):
            copy_env = deepcopy(self._environment)
            next_state, reward, done, info = copy_env.step(a)
            self._children[a] = Node(network=self._network,
                                     environment=copy_env,
                                     state=next_state,
                                     parent=self,
                                     prior=p,
                                     scaler=self._scaler,
                                     reward=reward,
                                     done=done,
                                     depth=self.depth + 1)
        if cfg.use_dirichlet:
            self.add_exploration_noise()

    def select_child(self) -> int:
        """
        Select the child with the highest UCB-score
        :return: Index of child with highest UCB-score
        """
        ucb = []
        for a in cfg.action_range:
            ucb.append(self._children[a].ucb_score)
        return int(np.argmax(ucb))

    def explore(self) -> None:
        """
        Searches the tree for not yet expanded nodes according to the UCB-score
        """
        if self.visit_count == 0 and not self.is_root:
            self.expand()
            self.visit_count = 1
            self._parent.rollout(self._value_sum)
        else:
            self._children[self.select_child()].explore()

    def add_exploration_noise(self) -> None:
        """
        Adds dirichlet exploration noise to the priors. Allows for smoother exploration
        """
        noise = np.random.dirichlet([cfg.dirichlet_alpha] * len(cfg.action_range))
        for a, n in zip(cfg.action_range, noise):
            self._children[a]._prior *= (1 - cfg.explore_fraction)
            self._children[a]._prior += n * cfg.explore_fraction

    def get_visit_count(self) -> Dict[int, int]:
        """
        Gets the visit count of all children nodes
        :return: dictionary with visit counts per child
        """
        child_dict = {}
        for idx in self._children:
            child_dict[idx] = self._children[idx].visit_count
        return child_dict

    def get_action(self) -> int:
        """
        Calculates which node was visited most often and returns it as the action index
        :return: Index of the next action to take according to visit count
        """
        visit_count_dict = self.get_visit_count()
        visit_count = [visit_count_dict[a] for a in visit_count_dict]
        soft_visit_count = []
        for v in visit_count:
            soft_visit_count.append(v ** (1 / cfg.temperature) / sum(visit_count) ** (1 / cfg.temperature))
        soft_visit_count = np.array(soft_visit_count) / np.sum(soft_visit_count)
        return int(np.argmax(soft_visit_count))

    def get_priors(self) -> np.array:
        """
        Calculates the priors from the visit counts of all childs
        :return: Distribution over all childs
        """
        visit_count_dict = self.get_visit_count()
        visit_count = np.array([visit_count_dict[a] for a in visit_count_dict])
        sum_visits = sum(visit_count)
        return visit_count / sum_visits
