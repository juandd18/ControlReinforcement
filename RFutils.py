from unityagents import UnityEnvironment
import numpy as np
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, VecEnv
#copy utils code from  ttps://github.com/ShangtongZhang/DeepRL.git
from utils import *



class Task:
    def __init__(self,
                 name,
                 env,
                 num_envs=1,
                 single_process=True,
                 log_dir=None,
                 episode_life=True,
                 seed=np.random.randint(int(1e5))):
        if log_dir is not None:
            mkdir(log_dir)
        envs = [UnityEnvironment(file_name=name) for i in range(num_envs)]
        #multiple envs various agents
        Wrapper = SubprocVecEnv
        self.env = Wrapper(envs)
        self.name = name
        self.observation_space = self.env.observation_space

        #continuous state space
        self.state_dim = int(np.prod(self.env.observation_space.shape))

        self.action_space = self.env.action_space
        self.action_dim = self.action_space.shape[0]

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        #continuous state space
        actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)