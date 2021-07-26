"""
Example showing how to wrap the iGibson class using ray for rllib.
Multiple environments are only supported on Linux. If issues arise, please ensure torch/numpy
are installed *without* MKL support.

This example requires ray to be installed with rllib support, and pytorch to be installed:
    `pip install torch "ray[rllib]"`

Note: rllib only supports a single observation modality:


# Stable-baselines PPO policy
https://github.com/DLR-RM/stable-baselines3/blob/2fa06ae8d24662a40a7c247dd96625bebf182dce/stable_baselines3/ppo/ppo.py#L15-L92

# Multi-input actor-critic policy 

https://github.com/DLR-RM/stable-baselines3/blob/2fa06ae8d24662a40a7c247dd96625bebf182dce/stable_baselines3/common/policies.py#L720-L773


# Ray rllib PPO policy

Common rllib config
https://github.com/ray-project/ray/blob/90a1667b2909e1ce453453c5707c00b1951c3dca/rllib/agents/trainer.py#L55-L472 

Overridden by PPO config
https://github.com/ray-project/ray/blob/90a1667b2909e1ce453453c5707c00b1951c3dca/rllib/agents/ppo/ppo.py#L36-L96


Possible differences

Value function clipping:
stable-baselines3: https://github.com/DLR-RM/stable-baselines3/blob/2fa06ae8d24662a40a7c247dd96625bebf182dce/stable_baselines3/ppo/ppo.py#L224-L225

ray: https://github.com/ray-project/ray/blob/e69987bc966b9e7cfd8500918c977c05e5af6e34/rllib/agents/ppo/ppo_torch_policy.py#L81-L83

"""
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

class FCHead(nn.Module):

    def __init__(self, in_shape=232, out_shape=64):
        super(FCHead, self).__init__()
        self.fc1 = nn.Linear(in_shape, 64, bias=True)
        self.fc2 = nn.Linear(64, out_shape, bias=True)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

class FC(nn.Module):

    def __init__(self, in_shape=64, out_shape=2):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(in_shape, out_shape, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        return x

class iGibsonPPOModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.share_head = FCHead(in_shape=232, out_shape=64)
        self.value_head = FC(in_shape=64, out_shape=1)
        self.action_head = FC(in_shape=64, out_shape=2)

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

        shared = self.share_head(policy_input)
        self._value_out = torch.flatten(self.value_head(shared))
        action_out = self.action_head(shared)

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
                        default='ray_example',
                        help='which mode for simulation (default: headless)')
    parser.add_argument('--mode',
                        '-m',
                        choices=['headless', 'gui', 'iggui'],
                        default='headless',
                        help='which mode for simulation (default: headless)')
    
    args = parser.parse_args()

    ModelCatalog.register_custom_model("iGibsonPPOModel", iGibsonPPOModel)
    register_env("iGibsonEnv", lambda c: iGibsonRayEnv(c))
    # See here for stable-baselines3 defaults
    #https://github.com/DLR-RM/stable-baselines3/blob/2fa06ae8d24662a40a7c247dd96625bebf182dce/stable_baselines3/ppo/ppo.py#L69-L91
    # Things currently missing/not confirmed to be equal
    # clip_range
    # clip_range_vf
    # not sure GAE/lambda matches
    # not sure if observation filtering matches
    config = {
        "env" : "iGibsonEnv",
        "env_config" : {
            "config_file": args.config,
            "mode": args.mode,
            # matches eric
            "action_timestep": 1.0 / 10.0,
            # matches eric
            "physics_timestep": 1.0 / 120.0
        },
        #ray specific
        "num_gpus": 1,
        # ray specific
        # "num_cpus_for_driver": 5,
        # "remote_worker_envs": True,
        # number of workers == number of environments, confirmed match
        "num_workers": 8,
        # ray specific
        "num_envs_per_worker": 1,
        # ray specific
        # "num_cpus_per_worker": 5,
        # equivalent to buffer size (num_envs * n_steps)
        "train_batch_size": 16384,
        # equivalent to learning rate
        "lr": 3e-4,
        # equivalent to n_steps
        "rollout_fragment_length": 2048,
        # equivalent to batch_size
        "sgd_minibatch_size": 64,
        # equivalent to num_epochs
        "num_sgd_iter": 10,
        "gamma": 0.99,
        "use_gae": True,
        # equivalent to GAE lambda
        "lambda": 0.95,
        # equivalent to clip_range
        "clip_param": 0.2,
        # -1 removes clipping, would be None in stable-baselines3
        "vf_clip_param": -1,
        "entropy_coeff": 0.0,
        "vf_loss_coeff": 0.5,
        # equivalent to max_grad_norm in stable-baselines3
        "grad_clip": 0.5,
        # normalization, none for now
        "kl_target": 0.01,
        # "observation_filter": "MeanStdFilter",
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
