import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import sympy as sp

# Define system parameters symbolically
m, g, l, I_p = sp.symbols('m g l I_p')

# Define system matrices symbolically
A = sp.Matrix([[0, 1, 0, 0],
              [0, 0, -m * g * l * l / I_p, 0],
              [0, 0, 0, 1],
              [0, 0, m * g * l / I_p, 0]])

B = sp.Matrix([[0],
              [(m * l**2 + I_p) / (m * I_p)],
              [0],
              [-l / I_p]])  # Corrected term

# Define cost matrices symbolically
Q = sp.diag(1, 1, 10, 1)
R = sp.Matrix([[1]])

# Solve Continuous-time Algebraic Riccati Equation (symbolically)
P = sp.Matrix(sp.simplify(sp.solve(A.T * P + P * A - P * B * R.inv() * B.T * P + Q, P)))

# Compute LQR gain symbolically
K = R.inv() * B.T * P

# Compute closed-loop system matrix symbolically
A_cl = A - B * K

# Compute eigenvalues of the closed-loop system symbolically
eigenvalues = A_cl.eigenvals()
print("Closed-loop eigenvalues (symbolic):", eigenvalues)

# Convert to numerical values for simulation
denom = m * I_p
A_num = np.array(A.subs({m: 1.0, g: 9.81, l: 1.0, I_p: 1.0})).astype(np.float64)
B_num = np.array(B.subs({m: 1.0, g: 9.81, l: 1.0, I_p: 1.0})).astype(np.float64)

# Simulate system response
dt = 0.01  # Time step
T = 5.0  # Total simulation time
num_steps = int(T / dt)

# Initial state (small perturbation)
x = np.array([[0.1], [0], [0.1], [0]])

# Store trajectory
time = np.linspace(0, T, num_steps)
states = np.zeros((num_steps, 4))

for i in range(num_steps):
    u = -K @ x  # Compute control input
    x_dot = A_num @ x + B_num @ u  # Compute state derivative
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
