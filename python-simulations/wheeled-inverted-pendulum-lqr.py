import numpy as np
import scipy.linalg

# Define system matrices (example values, replace as needed)
m = 1  # Mass
l = 0.5  # Length
g = 9.81  # Gravity
I_p = 1/3 * m * l**2  # Moment of inertia
A = np.array([[0, 1, 0, 0],
              [0, 0, -m * g * l**2 / I_p, 0],
              [0, 0, 0, 1],
              [0, 0, m * g * l / I_p, 0]])

B = np.array([[0],
              [m * l**2 + I_p / (m * I_p)],
              [0],
              [-l / I_p]])

# Define cost matrices (example values, replace as needed)
Q = np.diag([1, 1, 10, 1])  # Penalize theta more
R = np.array([[1]])  # Control effort cost

# Check controllability
Ctrb = np.hstack([B, A @ B, A @ A @ B, A @ A @ A @ B])
rank_Ctrb = np.linalg.matrix_rank(Ctrb)
print(f"Controllability matrix rank: {rank_Ctrb} (should be 4)")

# Check stabilizability (ensure unstable modes are controllable)
eigvals, _ = np.linalg.eig(A)
uncontrollable_eigs = [eig for eig in eigvals if eig.real > 0 and np.linalg.matrix_rank(np.hstack([A - eig*np.eye(4), B])) < 4]

if rank_Ctrb < 4:
    print("Warning: System may not be fully controllable!")
    if uncontrollable_eigs:
        print("System is NOT stabilizable. LQR may not work.")
    else:
        print("System is stabilizable. Proceeding with LQR.")
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        print("Solution P matrix:")
        print(P)
else:
    # Solve Continuous-time Algebraic Riccati Equation (CARE)
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    print("Solution P matrix:")
    print(P)

# Compute the LQR gain matrix K = R^(-1) B^T P
K = np.linalg.inv(R) @ B.T @ P
print("LQR gain matrix K:")
print(K)
