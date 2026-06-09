import os, pickle, pathlib
from enum import Enum
import numpy as np
import torch

import utils

class Logger:

    class Category(Enum):
        LOSS=1,
        REWARD=2,
        OTHER=3

    def __init__(self, use_tensorboard, use_files, network_save_interval, print_progress, save_loc, categories=[Category.LOSS, Category.REWARD, Category.OTHER]):
        self.use_files = use_files
        self.network_save_interval = network_save_interval
        self.save_network = network_save_interval > 0
        self.print_progress = print_progress

        self.save_loc = save_loc
        if os.path.isdir(self.save_loc):
            current_digit = 1
            current_check = f"{self.save_loc}_{current_digit}"
            while os.path.isdir(current_check):
                current_digit += 1
                current_check = f"{self.save_loc}_{current_digit}"
            self.save_loc = current_check

        self.tensorboard_writer = utils.create_tensorboard_writer(comment=f"_{save_loc}", flush_secs=5)
        self.use_tensorboard_logs = (self.tensorboard_writer is not None) and use_tensorboard

        self.vars = {}
        self.categories = categories

        if use_files:
            self._make_parent_dir()

        self.reward_record = -np.inf
        self.timesteps_completed = 0
        self.gradient_steps = 0
        self.episodes_completed = 0
        self.reward_history = []
        self.torch_network_dict = None
        self.saved_network_before = False

    def _make_parent_dir(self):
        pathlib.Path(f"{self.save_loc}").mkdir(parents=True, exist_ok=True)

    def _add_var_if_new(self, var_name):
        if var_name not in self.vars:
            self.vars[var_name] = []

    def _write_to_tensorboard(self, name, val, step):
        if self.use_tensorboard_logs:
            self.tensorboard_writer.add_scalar(name, val, step)
            self.tensorboard_writer.flush()

    def _save_vars_to_file(self):
        if self.use_files:
            for var_name, value in self.vars.items():
                if isinstance(value, list):
                    with open(f"{self.save_loc}/{var_name}.pkl", "wb") as f:
                        pickle.dump(value, f)
    
    def _save_network_to_file(self):
        if self.save_network and self.torch_network_dict is not None:
            self.saved_network_before = True
            if not os.path.isdir(self.save_loc):
                self._make_parent_dir()
            torch.save(self.torch_network_dict, f"{self.save_loc}/torch_network.pt")

    def timestep_complete(self):
        self.timesteps_completed += 1

    def gradient_step_complete(self, loss_names, loss_values):
        self.gradient_steps += 1
        if self.Category.LOSS in self.categories:
            for loss_idx in range(len(loss_names)):
                name = loss_names[loss_idx]
                val = loss_values[loss_idx]
                self._add_var_if_new(name)
                self.vars[name].append((val, self.gradient_steps))
                self._write_to_tensorboard(name, val, self.gradient_steps)
    
    def network_update(self, torch_network_dict):
        self.torch_network_dict = torch_network_dict
                
    def episode_complete(self, reward):
        self.episodes_completed += 1
        if self.print_progress:
            print(f"episode {self.episodes_completed}, timesteps {self.timesteps_completed}, reward: {reward}")
        self.reward_history.append(reward)
        if ((reward > self.reward_record) or 
            (not self.saved_network_before and self.save_network) or
            ((self.episodes_completed % self.network_save_interval == 0 and self.save_network))):
            if reward > self.reward_record:
                self.reward_record = reward
            self._save_vars_to_file()
            self._save_network_to_file()
        if self.Category.REWARD in self.categories:
            self._add_var_if_new("episodic_reward")
            self._add_var_if_new("mean_episodic_reward")
            self.vars["episodic_reward"].append((reward, self.episodes_completed))
            self.vars["mean_episodic_reward"].append((np.mean(self.reward_history[-100:]), self.timesteps_completed))
            self._write_to_tensorboard("episodic_reward", reward, self.episodes_completed)
            self._write_to_tensorboard("mean_episodic_reward", np.mean(self.reward_history[-100:]), self.timesteps_completed)

    def training_done(self):
        self._save_vars_to_file()
        self._save_network_to_file()