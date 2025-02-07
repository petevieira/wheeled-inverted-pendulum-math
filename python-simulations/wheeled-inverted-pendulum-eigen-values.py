import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import sympy as sp

# Define system parameters symbolically
m, g, l, I_p, Kp, Kd = sp.symbols('m g l I_p Kp Kd')

# Define system matrices symbolically
A = sp.Matrix([[0, 1, 0, 0],
              [0, 0, -m * g * l * l / I_p, 0],
              [0, 0, 0, 1],
              [0, 0, m * g * l / I_p, 0]])

B = sp.Matrix([[0],
              [(m * l**2 + I_p) / (m * I_p)],
              [0],
              [-l / I_p]])  # Corrected term

K = sp.Matrix([[0, 0, Kp, Kd]])

# Compute closed-loop system matrix symbolically
A_cl = A - B * K

# Compute eigenvalues of the closed-loop system symbolically
eigenvalues = A_cl.eigenvals()
print("Closed-loop eigenvalues (symbolic):", eigenvalues)
