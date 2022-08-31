import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import Union

# The helicopter simulation model developed here is based on previous own work:
# Taken from Gerrits, I. and Valencia Ibanez, S., 
# "AE4314 Helicopter Performance, Stability and Control: Final assignment 2022," Course assignment, 2022.
# Helicopter's parameters based on the Sikorsky UH-60A Blackhawk
aircraft_parameters = {
                       'M': 4945, 
                       'I_y': 54233,
                       'CD_S': 1.26,
                       'Omega': 27,
                       'R': 8.178,
                       'sigma': 0.08210,
                       'cl_alpha': 5.73,
                       'g': 9.81,
                       'rho': 1.225,
                       'gamma': 8.1936,
                       'h': 1.6
                        }

# Random initial conditions. Unused for actual environment simulation
initial_conditions = {
                      'x0': np.random.uniform(-10, 10),
                      'u0': np.random.uniform(-5, 5),
                      'z0': np.random.uniform(0, 10),
                      'w0': np.random.uniform(-5, 5),
                      'theta_f0': np.random.uniform(-np.pi/2, np.pi/2),
                      'q0': np.random.uniform(np.radians(-7), np.radians(7))
}


def fun(t: float, y: Union[list, np.ndarray], theta_0: float, theta_c: float,
         aircraft_parameters: dict) -> list:
    """
    Function that determines the 3-DoF helicopter's equations of motion
    Args:
        t (float): Current time
        y (Union[list, np.ndarray]): State vector [x, u, z, w, theta_f, q]^T
        theta_0 (float): Control action on collective pitch
        theta_c (float): Control action on cyclic pitch
        aircraft_parameters (dict): Dictionary with helicopter's parameters

    Returns:
        list: dy/dt vector
    """
    # Unpack the state vector y
    x, u, z, w, theta_f, q = y

    # Calculate velocity magnitude
    V = np.sqrt(u**2 + w**2)

    # Calculate the parameter alpha_c
    if u == 0:
        alpha_c = np.pi/2
    else:
        alpha_c = -np.arctan(w/u) + theta_c

        if u < 0:
            alpha_c += np.pi

    # Calculate the parameter mu    
    mu = (V/(aircraft_parameters['Omega']*aircraft_parameters['R']))*np.cos(alpha_c)
    # Calculate the parameter lambda_c
    lambda_c = (V/(aircraft_parameters['Omega']*aircraft_parameters['R']))*np.sin(alpha_c)

    def get_CT(lambda_i: float) -> tuple:
        """
        Calculate the thrust coefficient with the BEM and Glauert methods
        Args:
            lambda_i (float): Induced velocity

        Returns:
            tuple: Difference between the thrust coefficient from the two approaches, 
                   Glauert thrust coefficient, and the parameter a_1
        """
        # Calculate a_1
        a_1 = (
                -16*q/(aircraft_parameters['gamma']*aircraft_parameters['Omega'])
                + 8*mu*theta_0/3 - 2*mu*(lambda_c + lambda_i)
              )/(1 - mu**2/2)
        
        # Calculate thrust coefficient with BEM
        C_T_elem = (
                    (aircraft_parameters['cl_alpha']*aircraft_parameters['sigma']/4)*
                    ((2/3)*theta_0*(1+3*mu**2/2) - (lambda_c + lambda_i))
                   )

        # Calculate thrust coefficient with Glauert's method        
        C_T_Glauert = (
                        2*lambda_i*
                        np.sqrt(
                            (V*np.cos(alpha_c - a_1)/(aircraft_parameters['Omega']*aircraft_parameters['R']))**2 
                            + (V*np.sin(alpha_c - a_1)/(aircraft_parameters['Omega']*aircraft_parameters['R'])
                            + lambda_i)**2
                        )
                      )
        
        # Calculate difference between BEM and Glauert thrust coefficients
        f = C_T_elem - C_T_Glauert

        return f, C_T_Glauert, a_1
    
    # Determine the correct induced velocity lambda_i by iteratively making
    # C_T_Glauert and C_T_elem equal
    lambda_i = fsolve(lambda x: get_CT(x)[0], 0.1, xtol=1e-4)[0]

    # Get the correct thrust coefficient and a_1 parameter with the lambda_i calculated above
    C_T, a_1 = get_CT(lambda_i)[1:]
    # Calculate dimensional thrust
    T = C_T*aircraft_parameters['rho']*(aircraft_parameters['Omega']*aircraft_parameters['R']
                                        )**2*np.pi*aircraft_parameters['R']**2
    # Calculate dimensional drag
    D = aircraft_parameters['CD_S']*0.5*aircraft_parameters['rho']*V**2
    
    # Longitudinal acceleration
    u_dot = (
             -aircraft_parameters['g']*np.sin(theta_f) - (D*u)/(aircraft_parameters['M']*V)
             +(T/aircraft_parameters['M'])*np.sin(theta_c - a_1) - q*w
            )
    # Vertical acceleration
    w_dot = (
             aircraft_parameters['g']*np.cos(theta_f) - (D*w)/(aircraft_parameters['M']*V)
             -(T/aircraft_parameters['M'])*np.cos(theta_c - a_1) + q*u
            )
    
    # Angular (pitch) acceleration
    q_dot = -(T/aircraft_parameters['I_y'])*aircraft_parameters['h']*np.sin(theta_c - a_1)

    # Vector of derivatives:
    #       x_dot, u_dot, z_dot, w_dot, theta_f_dot, q_dot
    y_dot = [u, u_dot, w, w_dot, q, q_dot]

    return y_dot

def take_step(theta_0: float, theta_c: float, y_previous: Union[list, np.ndarray], 
             step_size: float=0.01) -> np.ndarray:
    """
    Take an integration step given an action and a previous state
    Args:
        theta_0 (float): Collective control action
        theta_c (float): Cyclic control action
        y_previous (Union[list, np.ndarray]): Previous state: x, u, z, w, theta_f, q, err_theta_f
        step_size (float, optional): Integration step size. Defaults to 0.01.

    Returns:
        np.ndarray: New state following action and old state
    """
    # Remove err_theta_f part of the state (only used for PID)
    y_previous = y_previous[:-1]
    # Integrate the differential equations. BDF method used because the
    # problem may be stiff
    sol = solve_ivp(fun, t_span=(0, step_size), y0=y_previous, 
                    args=[theta_0, theta_c, aircraft_parameters],
                    max_step=step_size, method='BDF')
    
    # Get new state from integration step
    x, u, z, w, theta_f, q = sol.y[:, -1]

    # Return the new state (including theta_f error step for PID)
    return np.array([x, u, z, w, theta_f, q, theta_f - (-0.026*(0-x) + 9e-2*u)])

def main():
    initial_conditions = {
                      'x0': np.random.uniform(-10, 0),
                      'u0': np.random.uniform(0, 10),
                      'z0': 0,
                      'w0': 0,
                      'theta_f0': 0,
                      'q0': 0

}
    dt = 0.01
    y0 = np.array(list(initial_conditions.values()))
    theta_0, theta_c = 0, 0
    for i in range(int(60/dt)):

        sol = solve_ivp(fun, t_span=(0, dt), y0=y0, args=[theta_0, theta_c, aircraft_parameters],
                max_step = dt)

        try:
            history = np.vstack((history, sol.y[:, -1]))
        except:
            history = sol.y[:, -1]
        
        y0 = sol.y[:, -1]

        x, u, z, w, theta_f, q = y0

        theta_f_wish = -0.026*(0-x) + 9e-2*u

        err_theta_f = theta_f - theta_f_wish

        try:
            theta_f_wish_history = np.append(theta_f_wish_history, err_theta_f)
        except:
            theta_f_wish_history = err_theta_f

        try:
            term = 10*q + 25*err_theta_f + 1*theta_f_wish_history.sum()*dt
            theta_c = term
        except IndexError:
            theta_c = 10*q + 25*err_theta_f
        
        try:
            theta_0 = 0.5*z + 0.5*w + history[:, 2].sum()*dt
        except IndexError:
            theta_0 = 0.5*z + 0.5*w

        print(x, z)

    print(history.shape)
    plt.plot(history[:, 0], history[:, 2])
    plt.scatter(history[0, 0], history[0, 2], marker='>')
    plt.axis('equal')
    plt.show()
    plt.close()

    plt.plot(history[:, 4])
    plt.show()

if __name__ == '__main__':
    main()

    

