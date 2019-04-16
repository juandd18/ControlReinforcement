from unityagents import UnityEnvironment
import numpy as np
#copy utils code from  ttps://github.com/ShangtongZhang/DeepRL.git
import torch
import torch.nn as nn
import torch.nn.functional as F


def collect_episodes(model, env, brain_name, states_ini, done, n_steps,config):
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

    config = config
    storage = Storage(config.rollout_length)
    states = states_ini

    while true:
        
        for _ in range(config.rollout_length):
            model.eval()
            with torch.no_grad():
                state = torch.from_numpy(states).float().to(device)
                prediction = model.forward(state)
            
            env_info = env.step(prediction.a.cpu().data.numpy())[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            done = env_info.local_done
        
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),'m': tensor(1 - dones).unsqueeze(-1)})

            states = next_states
        
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

        states = state_ini

        model.eval()
        with torch.no_grad():
            state = torch.from_numpy(states).float().to(device)
            prediction = model.forward(state)
        model.train()
        
        env_info = env.step(prediction.a.cpu().data.numpy())[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        done = env_info.local_done

        states = next_states
        state_ini = next_states
        
        if done == True:
            break
        
    return storage

def learn(storage, network, optimizer,config):
    
    log_prob, value, returns, advantages, entropy = storage.cat(['log_pi_a', 'v', 'ret', 'adv', 'ent'])
    policy_loss = -(log_prob * advantages).mean()
    value_loss = 0.5 * (returns - value).pow(2).mean()
    entropy_loss = entropy.mean()

    optimizer.zero_grad()
    (policy_loss - config.entropy_weight * entropy_loss +
        config.value_loss_weight * value_loss).backward()
    nn.utils.clip_grad_norm_(network.parameters(), config.gradient_clip)
    optimizer.step()
