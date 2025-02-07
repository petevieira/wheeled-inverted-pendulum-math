import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# PID Controller for tau_r
def pid_control(theta, theta_dot, Kp, Ki, Kd, integral, prev_error, dt):
    error = -theta  # Want to stabilize at theta = 0
    integral += np.clip(error * dt, -integrator_max, integrator_max)  # Anti-windup
    derivative = (error - prev_error) / dt
    control_signal = Kp * error + Ki * integral + Kd * derivative
    control_signal = np.clip(control_signal, -tau_r_max, tau_r_max)  # Clip the control signal
    return -control_signal, integral, error

def inverted_pendulum_dynamics(X, m, l, I_p, g, Kp, Ki, Kd, integral, prev_error, dt):
    r, r_dot, theta, theta_dot = X

    # Simulate ground contact
    if (abs(theta) > np.pi/2):
        return [0, 0, 0, 0], integral, prev_error, 0, integral  # Pendulum has fallen over

    # Compute mass matrix determinant
    det_M = m * I_p + m**2 * l**2 * (1 - np.cos(theta)**2)
    if (det_M < 1e-6):
        print('Singular mass matrix')
        return [0, 0, 0, 0], integral, prev_error, 0, integral  # Singular mass matrix

    # PID control for tau_r
    tau_r, integral, prev_error = pid_control(theta, theta_dot, Kp, Ki, Kd, integral, prev_error, dt)
    tau_theta = 0  # No direct torque on the pendulum

    # Compute accelerations
    r_ddot = ((m * l**2 + I_p) * (m * l * theta_dot**2 * np.sin(theta) + tau_r)
              - m * l * np.cos(theta) * (m * g * l * np.sin(theta) + tau_theta)) / det_M

    theta_ddot = (-m * l * np.cos(theta) * (m * l * theta_dot**2 * np.sin(theta) + tau_r)
                  + m * (m * g * l * np.sin(theta) + tau_theta)) / det_M

    return [r_dot, r_ddot, theta_dot, theta_ddot], integral, prev_error, tau_r, integral

# Simulation parameters
time_span = (0, 20)  # Simulate for 10 seconds
initial_conditions = [0, 0, np.pi/60, 0]  # [r, r_dot, theta, theta_dot]

# System parameters
m = 1.0 # newton pendulum mass
l = 0.5 # meter long pendulum
I_p = 0.1 # kg m^2
g = 9.81 # m/s^2 of gravitational acceleration

# PID gains
Kp = 10 # Nm/rad
Ki = 0.0 # Nm/rad
Kd = 0.1 # Nm*s/rad
integral = 0
prev_error = 0

# Safety limits
global tau_r_max, integrator_max
tau_r_max = 4 # Maximum torque that can be applied to the axle based on the motor, in Nm
integrator_max = 0.5 # Maximum value of the integrator, in radians

# Time information
time_eval = np.linspace(time_span[0], time_span[1], 1000)
dt = time_eval[1] - time_eval[0]

def simulate():
    global integral, prev_error
    X = initial_conditions.copy()
    results = []
    for t in time_eval:
        dX, integral, prev_error, tau_r, integral = inverted_pendulum_dynamics(X, m, l, I_p, g, Kp, Ki, Kd, integral, prev_error, dt)
        X = [X[i] + dX[i] * dt for i in range(4)]
        results.append([t] + X + [tau_r, integral]) # [time, r, r_dot, theta, theta_dot, tau_r]
    return np.array(results)

simulation = simulate()

# Plot results
plt.figure(figsize=(10, 5))
plt.subplot(4, 1, 1)
plt.plot(simulation[:, 0], simulation[:, 3], label='Theta (rad)')
plt.xlabel('Time (s)')
plt.ylabel('Pendulum Angle')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(simulation[:, 0], simulation[:, 1], label='Axle Position (r)')
plt.xlabel('Time (s)')
plt.ylabel('Axle Position')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(simulation[:, 0], simulation[:, 5], label='Axle Acceleration (tau_r)')
plt.xlabel('Time (s)')
plt.ylabel('Axle Acceleration')
plt.legend()

# Integral of the error
plt.subplot(4, 1, 4)
plt.plot(simulation[:, 0], simulation[:, 6], label='Integral of the error')
plt.xlabel('Time (s)')
plt.ylabel('Integral of the error')
plt.legend()


plt.tight_layout()
plt.show()
