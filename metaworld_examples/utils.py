from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from garage.envs import GymEnv


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
