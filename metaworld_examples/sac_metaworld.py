#!/usr/bin/env python3
"""This is an example to train a task with SAC algorithm written in PyTorch."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from garage import wrap_experiment
from garage.envs import normalize
from garage.experiment import deterministic
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import SAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer

import argparse
import os.path as osp
from metaworld_examples.utils import make_metaworld_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='reach-v2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--base_log_dir', type=str, default='./data')
    args = parser.parse_args()

    base_log_dir = osp.expanduser(args.base_log_dir)

    @wrap_experiment(snapshot_mode='none',
                     log_dir=osp.join(base_log_dir, 'sac-' +
                                      args.env_name, str(args.seed)),
                     use_existing_dir=True)
    def sac_metaworld_batch(ctxt=None, env_name='reach-v2', seed=0):
        """Set up environment and algorithm and run the task.

        Args:
            ctxt (garage.experiment.ExperimentContext): The experiment
                configuration used by Trainer to create the snapshotter.
            seed (int): Used to seed the random number generator to produce
                determinism.

        """
        deterministic.set_seed(seed)
        trainer = Trainer(snapshot_config=ctxt)
        expl_env = normalize(make_metaworld_env(env_name, seed=seed))
        eval_env = normalize(make_metaworld_env(env_name, seed=seed))

        policy = TanhGaussianMLPPolicy(
            env_spec=expl_env.spec,
            hidden_sizes=[256, 256],
            hidden_nonlinearity=nn.ReLU,
            output_nonlinearity=None,
            min_std=np.exp(-20.),
            max_std=np.exp(2.),
        )

        qf1 = ContinuousMLPQFunction(env_spec=expl_env.spec,
                                     hidden_sizes=[256, 256],
                                     hidden_nonlinearity=F.relu)

        qf2 = ContinuousMLPQFunction(env_spec=expl_env.spec,
                                     hidden_sizes=[256, 256],
                                     hidden_nonlinearity=F.relu)

        replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

        sampler = LocalSampler(agents=policy,
                               envs=expl_env,
                               max_episode_length=expl_env.spec.max_episode_length,
                               worker_class=FragmentWorker)

        sac = SAC(env_spec=expl_env.spec,
                  eval_env=eval_env,
                  policy=policy,
                  qf1=qf1,
                  qf2=qf2,
                  sampler=sampler,
                  gradient_steps_per_itr=1000,
                  max_episode_length_eval=500,
                  replay_buffer=replay_buffer,
                  min_buffer_size=1e4,
                  target_update_tau=5e-3,
                  discount=0.99,
                  buffer_batch_size=256,
                  reward_scale=1.,
                  steps_per_epoch=1)

        if torch.cuda.is_available():
            set_gpu_mode(True)
        else:
            set_gpu_mode(False)
        sac.to()
        trainer.setup(algo=sac, env=expl_env)
        # total_timesteps = n_epochs * steps_per_epoch * sampler_batch_size = 1000 * 1 * 1000 = 1M
        trainer.train(n_epochs=1000, batch_size=1000)

    # s = np.random.randint(0, 1000)
    sac_metaworld_batch(env_name=args.env_name, seed=args.seed)
