import numpy as np

THREAD_NAME = "pid_0001"

# environment setting
action_len: int = 2
action_range: np.array = np.arange(action_len)

# Monte Carlo Tree Search and target settings
discount: float = 0.99
pb_c_base: int = 19652
pb_c_init: float = 1.25
use_dirichlet: bool = False
dirichlet_alpha: float = 0.25
explore_fraction: float = 0.5
explore_decay: float = 0.9
min_explore: float = 0.05
buffer_size: int = 5000
temperature: float = 0.5
prior_temp: float = 2

# network shapes
state_shape: tuple = (4,)
support_size = 10
bins = (support_size * 2 + 1)
out_len = 2

# network training setting
learning_rate: float = 1e-4
training_loops = 3
train_epochs: int = 20
batch_size: int = 32
weight_decay: float = 1e-4

# sample and iteration settings
complete_cycles: int = 2000
num_simulations: int = 20
max_moves: int = 200
num_episodes: int = 10
training_samples: int = max_moves * num_episodes * batch_size  # num_episodes * max_moves
max_search_depth: int = 5

drop_factor = 0.95
epoch_drop = 1
