from unityagents import UnityEnvironment
import numpy as np
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, VecEnv
#copy utils code from  ttps://github.com/ShangtongZhang/DeepRL.git
from utils import Config


class Task:
    def __init__(self,
                 name,
                 num_envs=1,
                 observation_space = 33,
                 state_dim = 660,
                 action_space = 4,
                 action_dim = 80,
                 single_process=True,
                 log_dir=None,
                 episode_life=True,
                 seed=np.random.randint(int(1e5))):
        if log_dir is not None:
            mkdir(log_dir)
            
        brain_name = ""
        envs = [num_envs]
        for i in range(num_envs):
            env = UnityEnvironment(file_name=name)
            envs[i] = env
            
        #multiple envs various agents
        Wrapper = VecEnv
        self.env = envs
        self.name = name
        self.observation_space = observation_space

        #continuous state space
        self.state_dim = state_dim

        self.action_space = action_space
        self.action_dim = action_dim

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        #continuous state space
        actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)