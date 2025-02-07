import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# Define system parameters
m = 1  # Mass
l = 0.5  # Length
g = 9.81  # Gravity
I_p = 1/3 * m * l**2  # Moment of inertia

# Define system matrices (updated A and B)
A = np.array([[0, 1, 0, 0],
              [0, 0, -m * g * l**2 / I_p, 0],
              [0, 0, 0, 1],
              [0, 0, m * g * l / I_p, 0]])

B = np.array([[0],
              [(m * l**2 + I_p) / (m * I_p)],
              [0],
              [-l / I_p]])  # Corrected term

# Define cost matrices
# Q: State cost matrix, where state is (r, r_dot, theta, theta_dot)
Q = np.diag([1, 5, 10, 1])  # Penalize theta more
# R: Control effort cost, where control effort is tau_r
R = np.array([[1]])  # Control effort cost

# Solve Continuous-time Algebraic Riccati Equation (CARE)
# A^T*P + P*A − P*B*R^−1*B^T*P + Q = 0
P = scipy.linalg.solve_continuous_are(A, B, Q, R)
print("Solution P matrix:")
print(P)

# Compute LQR gain matrix K = R^(-1) B^T P
K = np.linalg.inv(R) @ B.T @ P
print("LQR gain matrix K:")
print(K)

# Simulate system response
dt = 0.01  # Time step
T = 5.0  # Total simulation time
num_steps = int(T / dt)

# Initial state (small perturbation)
x = np.array([[0.1], [0], [0.1], [0]])

# Store trajectory
time = np.linspace(0, T, num_steps)
states = np.zeros((num_steps, 4))

states[0, :] = x.T
for i in range(num_steps):
    u = -K @ x  # Compute control input
    x_dot = A @ x + B @ u  # Compute state derivative
    x = x + x_dot * dt  # Euler integration
    states[i, :] = x.T

# Plot results
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, states[:, 0], label='Cart Position (r)')
plt.plot(time, states[:, 2], label='Pendulum Angle (theta)')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('State')
plt.title('LQR-Controlled Wheeled Inverted Pendulum')

plt.subplot(2, 1, 2)
plt.plot(time, states[:, 1], label='Cart Velocity (r_dot)')
plt.plot(time, states[:, 3], label='Pendulum Angular Velocity (theta_dot)')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Velocity')

plt.show()
