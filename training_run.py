import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import bottleneck as bn
import datetime
from helicopter_environment import HelicopterTrain
from training_utils import OUActionNoise, Buffer, update_target, get_actor, get_critic, policy

def main():
    """
    Single training run. Based on Hemant Singh's implementation of DDPG
    in Keras: https://keras.io/examples/rl/ddpg_pendulum/
    """

    # Create environment
    env = HelicopterTrain(mode='theta_0_PID', max_time=45, dt=0.01)

    # Get states, actions, and action space bounds
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    upper_bound, lower_bound = env.action_space.high, env.action_space.low

    ######################## Training parameters ####################################
    # Noise standard deviation
    std_dev = np.pi/16
    # Instantiate noise object
    ou_noise = OUActionNoise(mean=np.zeros(num_actions), std_deviation=float(std_dev)*np.ones(num_actions))

    # Create actor and critic models
    actor_model = get_actor(num_states, num_actions, upper_bound)
    critic_model = get_critic(num_states, num_actions)

    # Create target networks
    target_actor = get_actor(num_states, num_actions, upper_bound)
    target_critic = get_critic(num_states, num_actions)

    # Initially set the target networks' weights to those of the main networks 
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    # Optimizer learning rates for actor and critic models
    critic_lr = 0.001
    actor_lr = 0.001

    # Create the optimizers
    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    # Number of training episodes
    total_episodes = 200
    # Discount factor for future rewards
    gamma = 0.99
    # Target network update rate
    tau = 0.001

    # Instantiate a buffer object
    buffer = Buffer(num_states, num_actions, 1000000, 64)

    # Patience to end training prematurely if several episodes show no improvement
    patience = 50

    # Store the reward history of each episode
    ep_reward_list = []
    # Store the average reward history of the last few episodes
    avg_reward_list = []

    # Set counters for possible early termination
    patience_counter = 0
    performance_counter = 0

    # Store the network weights that give the best rewards
    actor_best_weights = actor_model.get_weights()
    critic_best_weights = critic_model.get_weights()

    # To measure training time
    t_ini = time.time()

    # Training loop for each episode
    for ep in range(total_episodes):
        
        # Time per episode
        t0 = time.time()

        # Reset environment
        prev_state = env.reset()
        # Initialize reward at 0
        episodic_reward = 0
        # Reset noise object
        ou_noise.reset()

        # Run each episode until terminal state is reached
        while True:
            # Make state a tf.Tensor
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            
            # Select action from actor model + noise
            action = policy(tf_prev_state, actor_model, ou_noise)
            # Get state and reward from environment
            state, reward, done, info = env.step(action)

            # Record (s, a, r, s') to buffer
            buffer.record((prev_state, action, reward, state))
            # Update episodic reward
            episodic_reward += reward
        
            # Learn offline from buffers' experiences
            buffer.learn(gamma, target_actor, target_critic, critic_model, critic_optimizer, actor_model, actor_optimizer)
            # Update target networks
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)

            # End the episode when `done` is True
            if done:
                break
            
            # Set previous state to current state and go back to start of loop
            prev_state = state

        # Add episodic reward to list
        ep_reward_list.append(episodic_reward)

        # Add the mean of last 40 episodes' rewards to list
        avg_reward = np.mean(ep_reward_list[-40:])
        avg_reward_list.append(avg_reward)

        # Print update on progress
        print(f"Episode * {ep} * Avg Reward is ==> {avg_reward:.2f} * Current reward is ==> {episodic_reward:.2f} * LR is ==> {actor_optimizer.learning_rate.numpy():.6f} * {time.time()-t0:.2f} s/it")
        
        # Update best weights if there is good performance
        if ep > 10 and all(ep_reward_list[-10:] >= max(avg_reward_list)):
            print('Updating best weights')
            actor_best_weights = actor_model.get_weights()
            critic_best_weights = critic_model.get_weights()
        
        # Plot trajectory (uncomment when in notebook)
        # if ep % 10 == 0:
        #     env.plot_trajectory()
        
        # Patience starts running out if no progress is made
        if ep > 50 and avg_reward_list[-1] < avg_reward_list[-2]:
            patience_counter += 1
        else:
            patience_counter = 0
        
        # Training can end early if 5 consecutive episodes have a reward of more than 50000
        if all(np.array(ep_reward_list[-5:]) > 50000):
            print("Terminated early (good performance)")
            break

        # Training can end early if no progress is being made
        if patience_counter >= patience:
            print("Terminated early (no patience left)")
            break
        
        # Show plot of training progress (uncomment when in notebook)
        # if ep % 50 == 0 and ep > 0:
        #     plt.plot(avg_reward_list)
        #     plt.xlabel("Episode")
        #     plt.ylabel("Avg. Epsiodic Reward")
        #     plt.show()
        

    print(f'Training took {time.time()-t_ini:.2f} seconds')

    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()

    # Get current datetime
    now = str(datetime.datetime.now()).split('.')[0].replace(':', '').replace(' ', '_')
    # Save reward histories
    np.savetxt(f'rewards/ep_reward_history_{now}.txt', np.array(ep_reward_list))
    np.savetxt(f'rewards/avg_reward_history_{now}.txt', np.array(avg_reward_list))

    # Save actor and critic networks
    actor_model.save(f'actors/actor_{now}')
    critic_model.save(f'critics/critic_{now}')

if __name__ == '__main__':
    main()