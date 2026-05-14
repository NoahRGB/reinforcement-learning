import os, pickle, pathlib
from enum import Enum

import torch

from utils import create_tensorboard_writer

class Logger:

    class VarType(Enum):
        SCALAR = 1,
        TEXT = 2,

    def __init__(self, use_normal_logs, use_tensorboard_logs, parent_dir):
        self.use_normal_logs = use_normal_logs
        self.tensorboard_writer = create_tensorboard_writer(comment=f"_{parent_dir}", flush_secs=5)
        self.use_tensorboard_logs = (self.tensorboard_writer is not None) and use_tensorboard_logs
        self.vars = {}

        self.parent_dir = parent_dir
        if os.path.isdir(self.parent_dir):
            current_digit = 1
            current_check = f"{self.parent_dir}_{current_digit}"
            while os.path.isdir(current_check):
                current_digit += 1
                current_check = f"{self.parent_dir}_{current_digit}"
            self.parent_dir = current_check

        if use_normal_logs:
            self.make_parent_dir()
    
    def make_parent_dir(self):
        pathlib.Path(f"{self.parent_dir}").mkdir(parents=True, exist_ok=True)

    def log(self, var_name, value, step=None):

        if self.use_normal_logs:
            var_type = self.VarType.TEXT if isinstance(value, str) else self.VarType.SCALAR

            if var_type == self.VarType.SCALAR:
                if var_name not in self.vars: self.vars[var_name] = []
                self.vars[var_name].append(value)
            elif var_type == self.VarType.TEXT:
                self.vars[var_name] = value
            
        if self.use_tensorboard_logs and step is not None:
            self.tensorboard_writer.add_scalar(var_name, value, step)
            self.tensorboard_writer.flush()
    
    def save_logs(self):
        if self.use_normal_logs:
            for var_name, value in self.vars.items():
                if isinstance(value, list):
                    with open(f"{self.parent_dir}/{var_name}.pkl", "wb") as f:
                        pickle.dump(value, f)
                elif isinstance(value, str):
                     with open(f"{self.parent_dir}/{var_name}.txt", "w") as f:
                        f.write(value)

    def save_torch(self, dict, name):
        if not os.path.isdir(self.parent_dir):
            self.make_parent_dir()
        if name.endswith(".pt"):
            name = name[:-3]
        torch.save(dict, f"{self.parent_dir}/{name}.pt")