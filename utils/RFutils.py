import numpy as np
#copy utils code from  ttps://github.com/ShangtongZhang/DeepRL.git
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float)
    x = torch.tensor(x, device=device, dtype=torch.float32 ,)
    return x

def collect_episodes(model, env, brain_name, states_ini, limit_iter, config):
    '''
    Params
    ======
        model (object): A2C model
        env (object): environment
        brain_name (string): brain name of environment
        init_states (n_process, state_size) (numpy): initial states for loop
        done (bool): tracker of episode end, default False
        n_steps (int): number of steps for reward collection
    Returns
    =======
        batch_s (T, n_process, state_size) (numpy): batch of states
        batch_a (T, n_process, action_size) (numpy): batch of actions
        batch_v_t (T, n_process) (numpy): batch of n-step rewards (aks target value)
        accu_rewards (n_process,) (numpy): accumulated rewards for process (being summed up on all process)
        init_states (n_process, state_size) (numpy): initial states for next batch
        done (bool): tracker of episode end
    '''
    storage = Storage(config.rollout_length)
    states = states_ini
    accu_rewards = np.zeros(states_ini.shape[0])
    device = config.device
    iter = 0

    while True:
        
        for _ in range(config.rollout_length):
            model.eval()
            with torch.no_grad():
                state = torch.from_numpy(states).float().to(device)
                prediction = model.forward(state)
            model.train()
            
            env_info = env.step(prediction['a'].cpu().data.numpy())[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            
            done = env_info.local_done
            done_step = np.array(done)
        
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),'m': tensor(1 - done_step).unsqueeze(-1)})
            states = next_states
        

        model.eval()
        with torch.no_grad():
            state = torch.from_numpy(states).float().to(device)
            prediction = model.forward(state)
        model.train()

        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = prediction['v'].detach()

        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        states = states_ini

        env_info = env.step(prediction['a'].cpu().data.numpy())[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        
        done =  env_info.local_done
        dones = np.array(done)
        rewards = np.array(rewards)
        

        accu_rewards += rewards
        states = next_states
        states_ini = next_states
        iter += 1
        
        if iter >= limit_iter:
            break

        if dones.any() == True:
            env_info = env.reset(train_mode=True)[brain_name]
            states_ini = env_info.vector_observations
        
    return storage,np.sum(accu_rewards),states_ini,dones

def learn(storage, network, optimizer,config):
    
    log_prob, value, returns, advantages, entropy = storage.cat(['log_pi_a', 'v', 'ret', 'adv', 'ent'])
    policy_loss = -(log_prob * advantages).mean()
    value_loss = 0.5 * (returns - value).pow(2).mean()
    entropy_loss = entropy.mean()
    loss = policy_loss - config.entropy_weight * entropy_loss + config.value_loss_weight * value_loss
    loss = Variable(loss, requires_grad = True)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(network.parameters(), config.gradient_clip)
    optimizer.step()

    return loss
