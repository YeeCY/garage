from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from garage.envs import GymEnv
from garage.trainer import Trainer, NotSetupError
from dowel import logger


def make_metaworld_env(env_name, seed=None):
    goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[
        env_name + "-goal-observable"]
    env = goal_observable_cls(seed=seed)

    # mt1 = metaworld.MT1(env_name, seed=seed)
    # env = mt1.train_classes[env_name]()
    # task = random.choice(mt1.train_tasks)
    # environment.set_task(task)
    # env = SingleMT1Wrapper(env, mt1.train_tasks)
    # env.set_task(mt1.train_tasks[0])

    # env = TaskNameWrapper(env, task_name=env_name)
    env = GymEnv(env, max_episode_length=env.max_path_length)
    # env = TaskNameWrapper(env, task_name=env_name)

    env.seed(seed)

    return env


class MetaWorldTrainer(Trainer):
    def save(self, epoch):
        """Save snapshot of current batch.

        Args:
            epoch (int): Epoch.

        Raises:
            NotSetupError: if save() is called before the trainer is set up.

        """
        if not self._has_setup:
            raise NotSetupError('Use setup() to setup trainer before saving.')

        logger.log('Saving snapshot...')

        params = dict()
        # Save arguments
        params['seed'] = self._seed
        params['train_args'] = self._train_args
        params['stats'] = self._stats

        # Save states
        # params['env'] = self._env
        # params['algo'] = self._algo
        # params['n_workers'] = self._n_workers
        # params['worker_class'] = self._worker_class
        # params['worker_args'] = self._worker_args
        params['replay_buffer'] = self._algo.replay_buffer

        self._snapshotter.save_itr_params(epoch, params)

        logger.log('Saved')
