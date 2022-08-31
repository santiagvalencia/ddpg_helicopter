from typing import Optional
from gym import Env
from gym.spaces import Box
import numpy as np
from helicopter_model import take_step
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Union

def get_random_state() -> dict:
    """
    Function to initialize the environment's state

    Returns:
        dict: Contains the randomly initialized state (variable names in keys, values in values)
    """
    random_dict = {
          'x': np.random.uniform(-6, -4), # x between -6 and -4 meters
          'u': np.random.uniform(-1, 4), # u between -1 and 4 m/s
          'z': np.random.uniform(-0.01, 0.01), # z between -0.01 and 0.01 m
          'w': np.random.uniform(-0.01, 0.01), # w between -0.01 and 0.01 m/s
          'theta_f': np.radians(np.random.uniform(-5, 5)), # theta_f between -5 and 5 deg
          'q': np.radians(np.random.uniform(-0.5, 0.5)) # q between -0.5 and 0.5 deg/s
    }

    # err_theta_f for PID purposes only
    random_dict['err_theta_f'] = random_dict['theta_f'] - (-0.026*(0-random_dict['x']) + 9e-2*random_dict['u'])

    # Return the dictionary
    return np.array(list(random_dict.values()))

def reward_func(state: Union[list, np.ndarray], action: Union[list, np.ndarray]) -> float:
    """
    Defines the reward function for a given state and action

    Args:
        state (Union[list, np.ndarray]): The input state
        action (Union[list, np.ndarray]): The input action

    Returns:
        float: The calculated reward
    """

    # Unpack the state
    x, u, z, w, theta_f, q, err_theta_f = state
    # Unpack the action
    theta_0, theta_c = action

    # Initial reward is 1 for every step taken
    reward = 1
    
    def penalty_parabola(a, b):
        """
        Quadratic reward helper function
        parabola through (-a, 0), (0, b), (0, a)
        a: when reward = 0
        b: max reward 
        """
        return lambda v: b - b*v**2/a**2 

    # Define quadratic rewards
    reward += np.clip(penalty_parabola(10, 1/16)(np.degrees(q)), 0, 1)
    reward += np.clip(penalty_parabola(10, 1/8)(np.degrees(theta_f)), 0, 1)
    reward += np.clip(penalty_parabola(5, 1)(x), -1, 1)
    reward += np.clip(penalty_parabola(5, 1)(z), -1, 1)
    reward += np.clip(penalty_parabola(10, 1)(u), -1, 1)

    # Additional reward if going towards the target
    if x*u < 0:
        reward += 1
    else:
        reward -= 0.5
    
    # Additional reward if position within 2x2 m square around origin
    if abs(x) < 1 and abs(z) < 1:
        reward += np.clip(penalty_parabola(5, 1)(u), 0, 1)
    
    # Additional x- and u-based rewards
    if abs(x) < 4:
        reward += 1
        if abs(x) < 1 and abs(u) < 1:
            reward += 1
            if abs(x) < 0.1:
                reward += 5

    # Controller rewards
    reward += np.clip(penalty_parabola(60, 1)(np.degrees(theta_c)), -4, 1)
    reward += np.clip(penalty_parabola(60, 1)(np.degrees(theta_0)), -4, 1)

    # Penalize q in hopes of stabilizing oscillations
    reward += np.clip(penalty_parabola(10, 1)(np.degrees(q)), -2, 1)

    return reward

class HelicopterTrain(Env):
    """
    OpenAI Gym Env subclass that makes an environment to train a 
    3-degree-of-freedom helicopter via reinforcement learning.
    """

    def __init__(self, render_mode: Optional[str]=None, mode: str='full', 
    max_time: Union[float, int]=10, dt: float=0.01):
        """
        Initialize instance of HelicopterTrain environment

        Args:
            render_mode (Optional[str], optional): Needed for Env superclass. Not really used here. Defaults to None.
            mode (str, optional): Mode for environment creation. Options are:
                                  'full': Both theta_0 and theta_c are controlled by agent
                                  'theta_c_PID': theta_c is PID-controlled, theta_0 is controlled by agent
                                  'theta_0_PID': theta_0 is PID-controlled, theta_c is controlled by agent
                                  'full_PID': No agent, environment is fully PID-controlled
                                 Defaults to 'full'.
            max_time (Union[float, int], optional): Maximum simulation time in seconds. Defaults to 10.
            dt (float, optional): Time step used for differential equation integration. Defaults to 0.01.
        """

        # Define the observation space with gym's continuous Box
        self.observation_space = Box(
                            low=np.array([-10, -10, -10, -10, -np.pi/2, np.radians(-7), -np.inf], dtype=np.float64),
                            high=np.array([10, 10, 10, 10, np.pi/2, np.radians(7), np.inf], dtype=np.float64),
                            dtype=np.float64
                            )
        
        # Save the environment mode
        self.mode = mode

        # Set size of action space
        if mode == 'full':
            action_size = 2
        else:
            action_size = 1
        
        # Define the action space with gym's continuous Box
        self.action_space = Box(low=-np.pi/2, 
                                high=np.pi/2, shape=(action_size,), 
                                dtype=np.float64)

        # Initialize state
        self.state = get_random_state()

        # Initialize a history to keep track of trajectory
        self.history = self.state

        # Initialize histories to keep track of actions and rewards
        self.action_history = []
        self.reward_history = []

        # Set the maximum number of rounds
        self.max_rounds = int(max_time/dt)
        # Initialize rounds countdown at max_rounds
        self.rounds = self.max_rounds

        # Save maximum time and time step
        self.max_time = max_time
        self.dt = dt

        # Start a collected reward tracker
        self.collected_reward = 0

    def step(self, action: Union[list, np.ndarray]) -> tuple:
        """
        Take one step given an action

        Args:
            action (Union[list, np.ndarray]): The action to follow (control input)

        Returns:
            tuple: observation, reward, done, info tuple following gym's Env format
        """
        # Initialize done indicator and (unused) info dictionary
        done = False
        info = {}
        # Start reward at 0
        rw = 0
        # Remove one of the rounds available
        self.rounds -= 1

        # Unpack current state
        x, u, z, w, theta_f, q, err_theta_f = self.state
        
        # If mode is 'full', expected action is theta_0, theta_c
        if self.mode == 'full':
            theta_0, theta_c = action
            # A step is taken with the input action
            obs = take_step(theta_0, theta_c, self.state, step_size=self.dt)
        
        # If mode is 'theta_c_PID', expected action is theta_0 only
        elif self.mode == 'theta_c_PID':
            # theta_c is calculated from a manually tuned PID controller
            try:
                # Try to integrate if there is a history to do so
                theta_c = np.clip(10*q + 25*err_theta_f + 1*self.history[:, 6].sum()*self.dt, -np.pi/2, np.pi/2)
            except IndexError:
                # Otherwise skip the integration and only use the 'P' and 'D' parts of the PID 
                theta_c = np.clip(10*q + 25*err_theta_f, -np.pi/2, np.pi/2)
            
            # theta_0 is the input action
            theta_0 = action
            # Take a step with the input theta_0 + the PID-determined theta_c
            obs = take_step(theta_0, theta_c, self.state, step_size=self.dt)
        
        # If mode is 'theta_0_PID', expected action is theta_c only
        elif self.mode == 'theta_0_PID':
            # theta_0 is calculated from a manually tuned PID comtroller
            try:
                # Try to integrate if there is a history to do so
                theta_0 = np.clip(0.5*z + 0.5*w + self.history[:, 2].sum()*self.dt, -np.pi/2, np.pi/2)
            except IndexError:
                # Otherwise skip the integration and only use the 'P' and 'D' parts of the PID
                theta_0 = np.clip(0.5*z + 0.5*w, -np.pi/2, np.pi/2)

            # theta_c is the input action
            theta_c = action
            obs = take_step(theta_0, theta_c, self.state, step_size=self.dt)

        # If mode is 'full_PID', the input action is not used at all
        elif self.mode == 'full_PID':
            # Determine theta_0 and theta_c according to a manually tuned PID controller
            try:
                theta_0 = np.clip(0.5*z + 0.5*w + self.history[:, 2].sum()*self.dt, -np.pi/2, np.pi/2)
                theta_c = np.clip(10*q + 25*err_theta_f + 1*self.history[:, 6].sum()*self.dt, -np.pi/2, np.pi/2)
            except IndexError:
                theta_0 = np.clip(0.5*z + 0.5*w, -np.pi/2, np.pi/2)
                theta_c = np.clip(10*q + 25*err_theta_f, -np.pi/2, np.pi/2)
            # Take a step with the PID's action
            obs = take_step(theta_0, theta_c, self.state, step_size=self.dt)        
        
        # Standardize the action as a list with two entries for later processing
        action = np.array([theta_0, theta_c])
        
        # Add current action to action history
        self.action_history.append(action)

        # Update state
        self.state = obs

        # Add current state to history
        self.history = np.vstack((self.history, self.state))

        # Unpack state
        x, u, z, w, theta_f, q, err_theta_f = self.state

        # Find the reward of the current state
        rw += reward_func(self.state, action)

        # If the helicopter is out of bounds, state is terminal
        if abs(z) > 10 or abs(x) > 10:
            done = True

        # If no more rounds remain, state is terminal
        if self.rounds == 0:
            # Additional reward if final state is close to ideal of [0, 0, 0, 0, 0, 0]
            if np.sqrt(x**2 + z**2) < 0.1 and abs(theta_f) < np.radians(1) and np.sqrt(u**2 + w**2) < 0.1:
                rw += 1000
            done = True

        # Add current reward to history 
        self.reward_history.append(rw)

        # Return (observation, reward, done info) tuple following gym's Env format
        return obs, rw, done, info


    def render(self, action, rw):
        """
        Print out rounds remaining, actions, step reward, episodic reward, and position.
        Not really used.
        Args:
            action: Action taken
            rw: Corresponding reward
        """
        print(f'Round: {self.rounds}\nControllers: {np.degrees(action)}\nReward received: {rw}'
                +f'\nTotal reward: {self.collected_reward}\n (x, z): {self.state[0], self.state[2]}')
        print('================================================================================')

    def reset(self):
        """
        Reset the environment

        Returns:
            np.ndarray: New initial state 
        """
        # Reset state
        self.state = get_random_state()
        # Reset state history
        self.history = self.state
        # Reset rounds counter
        self.rounds = self.max_rounds
        # Reset episodic reward
        self.collected_reward = 0
        # Reset action history
        self.action_history = []
        # Reset reward history
        self.reward_history = []

        # Return the new initial state
        return self.state
    
    def plot_trajectory(self):
        """
        Make various plots of an episode's trajectory
        """
        fig, axs = plt.subplots(2, 5, figsize=(30, 10))

        # z vs. x plot
        axs[0, 0].plot(self.history[:, 0], self.history[:, 2])
        axs[0, 0].add_patch(Rectangle((-1, -1), 2, 2, fc='none', ec='black',
                        linestyle='dashed', label='Desired performance'))
        axs[0, 0].scatter(0, 0, color='green')
        axs[0, 0].scatter(self.history[0, 0], self.history[0, 2], color='blue', marker='>')
        axs[0, 0].set_xlim(-10, 10)
        axs[0, 0].set_ylim(-10, 10)

        # Actions vs. time plot
        theta_0 = [a[0] for a in self.action_history]
        theta_c = [a[1] for a in self.action_history]
        axs[0, 1].plot(np.degrees(theta_0), label=r'$\theta_0$')
        axs[0, 1].plot(np.degrees(theta_c), label=r'$\theta_c$')
        axs[0, 1].legend()
        axs[0, 1].set_title(f'{100*(1-self.rounds/self.max_rounds):.2f}%')

        # theta_f vs. time plot
        axs[0, 2].plot(np.degrees(self.history[:, 4]), label=r'$\theta_f$')
        axs[0, 2].set_title(r'$\theta_f$')
        axs[0, 2].legend()

        # q vs. time plot
        axs[0, 3].plot(np.degrees(self.history[:, 5]))
        axs[0, 3].set_title('q')

        # Speed (velocity magnitude) vs. time plot
        axs[0, 4].plot(np.sqrt(self.history[:, 3]**2 + self.history[:, 1]**2))
        axs[0, 4].set_title('V')

        # x vs. time plot
        axs[1, 0].plot(self.history[:, 0])
        axs[1, 0].set_title('x')

        # z vs. time plot
        axs[1, 1].plot(self.history[:, 2])
        axs[1, 1].set_title('z')

        # u vs. time plot
        axs[1, 2].plot(self.history[:, 1])
        axs[1, 2].set_title('u')

        # w vs. time plot
        axs[1, 3].plot(self.history[:, 3])
        axs[1, 3].set_title('w')

        # Step reward vs. time plot
        axs[1, 4].plot(self.reward_history)
        axs[1, 4].set_title('Step rewards')        

        # Plot up to maximum rounds even if episode terminated before
        for ax in axs.flatten()[1:]:
            ax.set_xlim(0, self.max_rounds)
        plt.show()
        plt.close()

def main():
    env = HelicopterTrain(render_mode=None)

    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

    plt.plot(env.history[:, 0], env.history[:, 2])
    plt.gca().add_patch(Rectangle((5, 5.5), 2, 2, fc='none', ec='black',
                        linestyle='dashed', label='Desired performance'))
    plt.scatter(6, 6, color='green')
    plt.show()

if __name__ == '__main__':
    main()
