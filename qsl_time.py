"""
Quantum Speed Limit (QSL) Time Dynamics in Radical Pairs
========================================================

This script calculates and plots the Quantum Speed Limit time for radical pair 
dynamics under various Master Equations (Unitary, Haberkorn, Jones-Hore, Kominis).
It evaluates the time-averaged norm of the generators and the mixed-state fidelity 
to bound the evolution time of the system.
"""

import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Handle version differences in scipy's integration module
try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:
    from scipy.integrate import cumtrapz as cumulative_trapezoid

# =====================================================================
# 1. PHYSICAL CONSTANTS & PARAMETERS
# =====================================================================
GAMMA = 8.8e4     # Gyromagnetic ratio (rad / (s * uT))
B0_REF = 46.0     # External magnetic field magnitude (uT)

# Hyperfine coupling and recombination rates
A_z = 6 * GAMMA * B0_REF
kS = 1e4          # Singlet recombination rate (s^-1)
kT = 1e6          # Triplet recombination rate (s^-1)

# Simulation parameters
THETA_FIX = np.pi / 12      # Fixed angle for evaluation
T_MAX = 15.0e-6             # Maximum evolution time (15 microseconds)
N_STEPS = 5000              # Number of time steps
t_eval = np.linspace(0, T_MAX, N_STEPS)

# =====================================================================
# 2. SPIN OPERATORS & BASIS PROJECTIONS
# =====================================================================
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
eye2 = np.eye(2, dtype=complex)

def kronecker_product(s_D, s_A, i_N): 
    """Helper to generate 3-spin Kronecker products."""
    return np.kron(s_D, np.kron(s_A, i_N))

# Spin operators for Donor (D), Acceptor (A), and Nucleus (N)
sDx = 0.5 * kronecker_product(sigma_x, eye2, eye2)
sDy = 0.5 * kronecker_product(sigma_y, eye2, eye2)
sDz = 0.5 * kronecker_product(sigma_z, eye2, eye2)

sAx = 0.5 * kronecker_product(eye2, sigma_x, eye2)
sAy = 0.5 * kronecker_product(eye2, sigma_y, eye2)
sAz = 0.5 * kronecker_product(eye2, sigma_z, eye2)

Ix  = 0.5 * kronecker_product(eye2, eye2, sigma_x)
Iy  = 0.5 * kronecker_product(eye2, eye2, sigma_y)
Iz  = 0.5 * kronecker_product(eye2, eye2, sigma_z)

# Singlet (QS) and Triplet (QT) Projectors
QS = 0.25 * np.eye(8, dtype=complex) - (sDx@sAx + sDy@sAy + sDz@sAz)
QT = np.eye(8, dtype=complex) - QS

# Triplet states for Kominis coherence calculations
state_T_plus = np.array([1, 0, 0, 0])
state_T_minus = np.array([0, 0, 0, 1])
state_T_0 = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0])

T_states = [
    np.kron(np.outer(state_T_plus, state_T_plus), eye2),
    np.kron(np.outer(state_T_minus, state_T_minus), eye2),
    np.kron(np.outer(state_T_0, state_T_0), eye2)
]

# Initial State: Electronic singlet + completely mixed nucleus
state_S = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0], dtype=complex)
rho_0 = np.kron(np.outer(state_S, state_S.conj()), 0.5 * eye2)

# Pre-compute sqrt(rho_0) to optimize the Bures angle fidelity metric later
sqrt_rho_0 = la.sqrtm(rho_0)

# =====================================================================
# 3. HAMILTONIAN & MASTER EQUATION GENERATORS
# =====================================================================
def get_hamiltonian(B0_mag, theta):
    """Constructs the system Hamiltonian with Zeeman and Hyperfine terms."""
    H_static = GAMMA * B0_mag * (np.sin(theta)*(sDx + sAx) + np.cos(theta)*(sDz + sAz))
    H_hyperfine = A_z * (sAz @ Iz)
    return H_static + H_hyperfine

def L_unitary(rho, H_mag):
    """Generator for standard unitary evolution (von Neumann)."""
    return -1j * (H_mag @ rho - rho @ H_mag)

def L_haberkorn(rho, H_mag):
    """Generator for the Haberkorn master equation."""
    unitary = L_unitary(rho, H_mag)
    recomb = - (kS / 2) * (QS @ rho + rho @ QS) - (kT / 2) * (QT @ rho + rho @ QT)
    return unitary + recomb

def L_hore(rho, H_mag):
    """Generator for the Jones-Hore master equation."""
    rho_ST = QS @ rho @ QT
    rho_TS = QT @ rho @ QS
    unitary = L_unitary(rho, H_mag)
    recomb = - (kS / 2) * (QS @ rho + rho @ QS) - (kT / 2) * (QT @ rho + rho @ QT)
    cross_term = - ((kS + kT) / 2) * (rho_ST + rho_TS)
    return unitary + recomb + cross_term

def _calculate_coherence_probability(rho):
    """Helper to extract coherence probability (p_coh) and trace for Kominis ME."""
    tr = np.real(np.trace(rho))
    if tr < 1e-15: 
        return 0.0, 0.0
        
    rho_ST = QS @ rho @ QT
    rho_TS = QT @ rho @ QS
    
    C_val = sum(np.sqrt(np.abs(np.trace(rho_ST @ P @ rho_TS))) for P in T_states)
    p_coh = np.clip((4.0 / 3.0) * C_val / tr, 0.0, 1.0)
    
    return p_coh, tr

def L_kominis(rho, H_mag):
    """Generator for the Kominis master equation."""
    p_coh, trace_rho = _calculate_coherence_probability(rho)
    if trace_rho < 1e-15: 
        return np.zeros_like(rho)
        
    L_uni = L_unitary(rho, H_mag)
    
    rho_ST = QS @ rho @ QT
    rho_TS = QT @ rho @ QS
    L_deph = L_uni - ((kS + kT) / 2.0) * (rho_ST + rho_TS)
        
    rho_SS = QS @ rho @ QS
    rho_TT = QT @ rho @ QT
    
    incoh_recomb = -(1.0 - p_coh) * (kS * rho_SS + kT * rho_TT)
    rate_S = kS * np.real(np.trace(rho_SS))
    rate_T = kT * np.real(np.trace(rho_TT))
    coh_recomb = - ((rate_S + rate_T) / trace_rho) * (p_coh * rho_SS + p_coh * rho_TT + rho_ST + rho_TS)
        
    return L_deph + incoh_recomb + coh_recomb

def me_wrapper(t, rho_flat, H_mag, L_func):
    """Reshapes flat state vectors for the ODE solver."""
    rho = rho_flat.reshape((8, 8))
    return L_func(rho, H_mag).flatten()

# =====================================================================
# 4. TIME-RESOLVED QSL CALCULATIONS
# =====================================================================
def operator_norm(A_mat):
    """Calculates the spectral norm of a matrix."""
    s = la.svd(A_mat, compute_uv=False)
    return float(np.max(s))

def mixed_state_fidelity(rho_tau):
    """Calculates fidelity amplitude Tr(sqrt(sqrt(rho0)*rho_tau*sqrt(rho0)))."""
    core = sqrt_rho_0 @ rho_tau @ sqrt_rho_0
    sqrt_core = la.sqrtm(core)
    F_amp = np.real(np.trace(sqrt_core))
    return np.clip(F_amp, 0.0, 1.0)

def compute_qsl_time_series(method_name):
    """Solves the IVP for a given generator and extracts QSL bounds."""
    H_mag = get_hamiltonian(B0_REF, THETA_FIX)
    
    # Map method string to the corresponding function
    generators = {
        "Unitary": L_unitary,
        "Haberkorn": L_haberkorn,
        "Jones-Hore": L_hore,
        "Kominis": L_kominis
    }
    L_func = generators[method_name]

    # Integrate the Master Equation
    y0 = rho_0.flatten()
    sol = solve_ivp(
        lambda t, r: me_wrapper(t, r, H_mag, L_func), 
        [0, T_MAX], y0, 
        t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-10, max_step=1e-8
    )
    
    rho_t = sol.y.T.reshape((N_STEPS, 8, 8))

    norm_L = np.zeros(N_STEPS)
    fid_t = np.zeros(N_STEPS)

    # Calculate instantaneous metrics
    for i in range(N_STEPS):
        norm_L[i] = operator_norm(L_func(rho_t[i], H_mag))
        fid_t[i] = mixed_state_fidelity(rho_t[i])

    # Integrate the cumulative norm of the generator over time
    integral_norm = cumulative_trapezoid(norm_L, t_eval, initial=0.0)

    Lambda_t = np.zeros(N_STEPS)
    tau_qsl = np.zeros(N_STEPS)

    # Compute time-averaged norm and final QSL bounds
    for i in range(1, N_STEPS):
        Lambda_t[i] = integral_norm[i] / t_eval[i]
        
        # QSL formulation: sin^2(Bures Angle) = 1 - Fidelity^2
        sin2_L = 1.0 - (fid_t[i]**2)
        tau_qsl[i] = sin2_L / Lambda_t[i] if Lambda_t[i] > 1e-18 else 0.0

    return tau_qsl, Lambda_t, fid_t

# =====================================================================
# 5. EXECUTION & PLOTTING
# =====================================================================
if __name__ == "__main__":
    print("Sweeping Time from 0 to 15 us (Fixed Theta).")
    
    results = {}
    methods = ["Unitary", "Haberkorn", "Jones-Hore", "Kominis"]
    
    for method in methods:
        print(f"Evaluating {method} dynamics.")
        t_qsl, lam, fid = compute_qsl_time_series(method)
        results[method] = {'tau_qsl': t_qsl, 'Lambda': lam, 'Fidelity': fid}

    # Convert seconds to microseconds for plotting
    t_us = t_eval * 1e6
    
    # Common plot styling mappings
    styles = {
        "Unitary": {'fmt': 'k:', 'label': 'Unitary evolution'},
        "Haberkorn": {'fmt': 'b--', 'label': 'Haberkorn ME'},
        "Jones-Hore": {'fmt': 'g-.', 'label': 'Jones-Hore ME'},
        "Kominis": {'fmt': 'r-', 'label': 'Kominis ME'}
    }

    # --- Figure 1: QSL Time ---
    plt.figure(figsize=(10, 6.5))
    plt.plot(t_us, t_us, 'purple', label='Actual driving time', linewidth=2.5)
    for method, style in styles.items():
        plt.plot(t_us, results[method]['tau_qsl'] * 1e6, style['fmt'], label=style['label'], linewidth=2.5)
    
    plt.xlabel(r'Evolution time, $t$ ($\mu$s)', fontsize=14)
    plt.ylabel(r'QSL time, $\tau_{\mathrm{QSL}}(t)$ ($\mu$s)', fontsize=14)
    plt.xlim(0, 15)
    plt.ylim(0, 15)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('QSL_Time_Figure_1_pi12.png', dpi=300)

    # --- Figure 2: Time-Averaged Norms ---
    plt.figure(figsize=(10, 6.5))
    for method, style in styles.items():
        plt.plot(t_us, results[method]['Lambda'] / 1e6, style['fmt'], label=style['label'], linewidth=2.5)

    plt.xlabel(r'Evolution time, $t$ ($\mu$s)', fontsize=14)
    plt.ylabel(r'Cumulative driving rate, $\Lambda_t$ ($\mu$s$^{-1}$)', fontsize=14)
    plt.xlim(0, 15)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('QSL_Time_Figure_2_pi12.png', dpi=300)

    # --- Figure 3: Fidelity ---
    plt.figure(figsize=(10, 6.5))
    for method, style in styles.items():
        plt.plot(t_us, results[method]['Fidelity'], style['fmt'], label=style['label'], linewidth=2.5)
    
    plt.xlabel(r'Evolution time, $t$ ($\mu$s)', fontsize=14)
    plt.ylabel(r'Fidelity, $F(\hat{\rho}_0, \hat{\rho}_t)$', fontsize=14)
    plt.xlim(0, 15)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('QSL_Time_Figure_3_pi12.png', dpi=300)