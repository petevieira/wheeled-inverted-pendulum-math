import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def inverted_pendulum_dynamics(t, X, m, l, I_p, g, tau_r, tau_theta):
    r, r_dot, theta, theta_dot = X
    # r = 0; r_dot = 0  # Assume the axle is fixed

    # Simulate ground contact
    if (abs(theta) > np.pi/2):
        return [0, 0, 0, 0] # Pendulum has fallen over

    # Compute mass matrix determinant
    det_M = m * I_p + m**2 * l**2 * (1 - np.cos(theta)**2)

    # Compute accelerations
    r_ddot = ((m * l**2 + I_p) * (m * l * theta_dot**2 * np.sin(theta) + tau_r)
              - m * l * np.cos(theta) * (m * g * l * np.sin(theta) + tau_theta)) / det_M

    theta_ddot = (-m * l * np.cos(theta) * (m * l * theta_dot**2 * np.sin(theta) + tau_r)
                  + m * (m * g * l * np.sin(theta) + tau_theta)) / det_M

    return [r_dot, r_ddot, theta_dot, theta_ddot]

# Simulation parameters
time_span = (0, 10)  # Simulate for 10 seconds
initial_conditions = [0, 0, np.pi/60, 0]  # [r, r_dot, theta, theta_dot]
X = initial_conditions
m, l, I_p, g = 1.0, 0.5, 0.1, 9.81  # System parameters
tau_r, tau_theta = 0, 0  # No control input for now

time_eval = np.linspace(time_span[0], time_span[1], 1000)

# Solve the equations
dynamics = lambda t, X: inverted_pendulum_dynamics(t, X, m, l, I_p, g, tau_r, tau_theta)
# solve_ivp(function_to_solve, time_span, initial_conditions, times_at_which_to_store_the_computed_solution, Runge-Kutta method)
sol = solve_ivp(dynamics, time_span, X, t_eval=time_eval, method='RK45')

# Plot results
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(sol.t, sol.y[2], label='Theta (rad)')
plt.xlabel('Time (s)')
plt.ylabel('Pendulum Angle')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(sol.t, sol.y[0], label='Axle Position (r)')
plt.xlabel('Time (s)')
plt.ylabel('Axle Position')
plt.legend()

plt.tight_layout()
plt.show()
