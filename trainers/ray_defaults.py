import argparse

from igibson.envs.igibson_env import iGibsonEnv

import os

from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.tune.registry import register_env

import ray
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import igibson

ray.init()

"""
MultiInputActorCriticPolicy(
  (features_extractor): CombinedExtractor(
    (extractors): ModuleDict(
      (task_obs): Flatten(start_dim=1, end_dim=-1)
      (scan): Flatten(start_dim=1, end_dim=-1)
    )
  )
  (mlp_extractor): MlpExtractor(
    (shared_net): Sequential()
    (policy_net): Sequential(
      (0): Linear(in_features=232, out_features=64, bias=True)
      (1): Tanh()
      (2): Linear(in_features=64, out_features=64, bias=True)
      (3): Tanh()
    )
    (value_net): Sequential(
      (0): Linear(in_features=232, out_features=64, bias=True)
      (1): Tanh()
      (2): Linear(in_features=64, out_features=64, bias=True)
      (3): Tanh()
    )
  )
  (action_net): Linear(in_features=64, out_features=2, bias=True)
  (value_net): Linear(in_features=64, out_features=1, bias=True)
)
"""

class FC(nn.Module):

    def __init__(self, in_shape=232, out_shape=2):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(in_shape, 64, bias=True)
        self.fc2 = nn.Linear(64, out_shape, bias=True)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

class iGibsonPPOModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.value_head = FC(in_shape=232, out_shape=1)
        self.action_head = FC(in_shape=232, out_shape=2)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        obs["task_obs"] = obs["task_obs"].float().flatten(start_dim=1)
        obs["scan"] = obs["scan"].float().flatten(start_dim=1)

        policy_input = torch.cat([
            obs["task_obs"],
            obs["scan"],
            ],
            dim=1
        )

        self._value_out = torch.flatten(self.value_head(policy_input))
        action_out = self.action_head(policy_input)

        return action_out, []

    def value_function(self):
        return self._value_out



class iGibsonRayEnv(iGibsonEnv):
    def __init__(self, env_config):
        super().__init__(
                config_file=env_config['config_file'],
                mode=env_config['mode'],
                action_timestep=env_config['action_timestep'],
                physics_timestep=env_config['physics_timestep'],
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        default='./configs/turtlebot_point_nav.yaml',
        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument(
        '--ray_mode',
        default="train",
        help='Whether to run ray in train or test mode')
    parser.add_argument(
        '--local_dir',
        default='/results',
        help='Directory where to save model logs and default checkpoints')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        choices=[None, "PROMPT"],
        help='Whether to resume the experiment. Note this does *not* restore the checkpoint, just re-uses the same config/log.')
    parser.add_argument(
        '--restore_checkpoint',
        default=None,
        help='Checkpoint to force restore')
    parser.add_argument('--exp_name',
                        default='ray_example_deeper_model',
                        help='which mode for simulation (default: headless)')
    parser.add_argument('--mode',
                        '-m',
                        choices=['headless', 'gui', 'iggui'],
                        default='headless',
                        help='which mode for simulation (default: headless)')
    
    args = parser.parse_args()

    ModelCatalog.register_custom_model("iGibsonPPOModel", iGibsonPPOModel)
    register_env("iGibsonEnv", lambda c: iGibsonRayEnv(c))
    config = {
        "env" : "iGibsonEnv",
        "env_config" : {
            "config_file": args.config,
            "mode": args.mode,
            "action_timestep": 1.0 / 10.0,
            "physics_timestep": 1.0 / 120.0
        },
        "num_gpus": 1,
        "num_cpus_for_driver": 5,
        "num_workers": 8,
        "num_envs_per_worker": 1,
        "num_cpus_per_worker": 5,
        # "sgd_minibatch_size": 256,
        "model": {
            "custom_model": "iGibsonPPOModel",
        },
        "framework": "torch"
    }
    stop={"training_iteration": 100000}
    if args.resume is not None:
        assert args.restore_checkpoint is not None, "Error: When resuming must provide explicit path to checkpoint"

    results = tune.run("PPO",
        config=config,
        verbose=2,
        restore=args.restore_checkpoint,
        name=args.exp_name,
        local_dir=args.local_dir,
        checkpoint_freq=100, 
        resume=args.resume
)
