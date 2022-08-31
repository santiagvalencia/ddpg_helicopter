from helicopter_environment import HelicopterTrain

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

import time

# from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
})

import bottleneck as bn

import datetime

from training_utils import OUActionNoise, Buffer, update_target, get_actor, get_critic, policy

def lr_sensitivity(lr, episodes=200):

    print(f'============ Learning rate = {lr:.2e} ===================')

    env = HelicopterTrain(mode='theta_0_PID', max_time=45, dt=0.01)

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    upper_bound, lower_bound = env.action_space.high, env.action_space.low

    print(num_states, num_actions, upper_bound, lower_bound)

    std_dev = np.pi/16
    ou_noise = OUActionNoise(mean=np.zeros(num_actions), std_deviation=float(std_dev) * np.ones(num_actions))

    actor_model = get_actor(num_states, num_actions, upper_bound)
    critic_model = get_critic(num_states, num_actions) # tf.keras.models.load_model('critic1/critic_25_08_2022_0007')

    target_actor = get_actor(num_states, num_actions, upper_bound)
    target_critic = get_critic(num_states, num_actions)

    # Making the weights equal initially
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    # Learning rate for actor-critic models
    critic_lr = lr
    actor_lr = lr

    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    total_episodes = episodes
    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    tau = 0.001

    buffer = Buffer(num_states, num_actions, 1000000, 64)

    patience = 300

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    big_history = []

    patience_counter = 0
    patience = 50

    performance_counter = 0

    actor_best_weights = actor_model.get_weights()
    critic_best_weights = critic_model.get_weights()

    t_ini = time.time()

    swa = False
    for ep in range(total_episodes):
        
        t0 = time.time()

        prev_state = env.reset()
        episodic_reward = 0
        
        ou_noise.reset()

        while True:
            
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            

            action = policy(tf_prev_state, actor_model, ou_noise)

            state, reward, done, info = env.step(action)

            buffer.record((prev_state, action, reward, state))
            episodic_reward += reward
        

            buffer.learn(gamma, target_actor, target_critic, critic_model, critic_optimizer, actor_model, actor_optimizer)
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)

            if done:
                break

            prev_state = state

        ep_reward_list.append(episodic_reward)
        
        big_history.append(env.history)

        avg_reward = np.mean(ep_reward_list[-40:])
        print(f"Episode * {ep} * Avg Reward is ==> {avg_reward:.2f} * Current reward is ==> {episodic_reward:.2f} * LR is ==> {actor_optimizer.learning_rate.numpy():.6f} * {time.time()-t0:.2f} s/it")
        avg_reward_list.append(avg_reward)
        
        if ep > 10 and all(ep_reward_list[-10:] >= max(avg_reward_list)):
            print('Updating best weights')
            actor_best_weights = actor_model.get_weights()
            critic_best_weights = critic_model.get_weights()
    


    print(f'Training took {time.time()-t_ini:.2f} seconds')

    return np.array(ep_reward_list), np.array(avg_reward_list)

def main():

    import pickle

    episodes = 200

    lrs = np.logspace(-5, -2, 4)

    rewards_dict = {}

    for lr in lrs:
        for i in range(3):

            ep_rewards, avg_rewards = lr_sensitivity(lr, episodes=episodes)

            if i == 0:
                rewards_dict[lr] = avg_rewards
            else:
                rewards_dict[lr] = np.vstack((rewards_dict[lr], avg_rewards))
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for lr, result in rewards_dict.items():
        mean_rewards = result.mean(axis=0)
        std_rewards = result.std(axis=0)

        plt.plot(mean_rewards, label=rf'$\lambda$ = {lr:.2E}')
        plt.fill_between(range(len(mean_rewards)), mean_rewards + std_rewards, mean_rewards-std_rewards, alpha=0.2)
    
    ax = plt.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlabel(r'Training episode', fontsize=16)
    ax.set_ylabel(r'Average rewards', fontsize=16)
    ax.legend(fontsize=15)
    
    plt.tight_layout()
    # plt.ylabel('Average rewards')
    # plt.xlabel('Training episode')
    plt.savefig('lr_sensitivity.pdf')
    # plt.show()
    plt.close()
    
if __name__ == '__main__':
    main()
    

