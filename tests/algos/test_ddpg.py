import os
os.environ['THEANO_FLAGS'] = 'device=cpu,mode=FAST_COMPILE,optimizer=None'
import unittest

from garage.algos import DDPG
from garage.envs.box2d import CartpoleEnv
from garage.exploration_strategies import OUStrategy
from garage.policies import DeterministicMLPPolicy
from garage.q_functions import ContinuousMLPQFunction
from garage.theano.envs import TheanoEnv


class TestDDPG(unittest.TestCase):
    def test_ddpg(self):
        env = TheanoEnv(CartpoleEnv())
        policy = DeterministicMLPPolicy(env.spec)
        qf = ContinuousMLPQFunction(env.spec)
        es = OUStrategy(env.spec)
        algo = DDPG(
            env=env,
            policy=policy,
            qf=qf,
            es=es,
            n_epochs=1,
            epoch_length=100,
            batch_size=32,
            min_pool_size=50,
            replay_pool_size=1000,
            eval_samples=100,
        )
        algo.train()
