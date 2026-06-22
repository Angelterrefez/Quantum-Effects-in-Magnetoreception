"""
Evolution Speed in Radical Pairs
===================================================================

This script calculates the instantaneous dynamical speed v(t) for 
radical pair systems. It evaluates it over time under different
master equations: Haberkorn, Jones-Hore, and Kominis.
"""

import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ==============================================================================
# 1. PHYSICAL CONSTANTS & PARAMETERS 
# ==============================================================================
GAMMA = 8.8e4     # Gyromagnetic ratio (rad / (s * uT))
B0_REF = 46.0     # External magnetic field magnitude (uT)

# Hyperfine coupling and recombination rates
A_z = 6 * GAMMA * B0_REF
kS = 1e4          # Singlet recombination rate (s^-1)
kT = 1e6          # Triplet recombination rate (s^-1)

# Simulation parameters
THETA_FIX = np.pi / 4  # Fixed angle for evaluation

# Evolve up to 5 lifetimes of the slowest recombination rate (15 microseconds).
T_MAX = 15e-6
N_TIME_STEPS = 500
t_eval = np.linspace(0, T_MAX, N_TIME_STEPS)

# ==============================================================================
# 2. SPIN OPERATORS & MEASUREMENT OBSERVABLES
# ==============================================================================
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
s_D_dot_s_A = sDx @ sAx + sDy @ sAy + sDz @ sAz
QS = 0.25 * np.eye(8, dtype=complex) - s_D_dot_s_A
QT = np.eye(8, dtype=complex) - QS

# Total spin operators for observables
S_total_x = sDx + sAx
S_total_y = sDy + sAy
S_total_z = sDz + sAz
S2_op = S_total_x @ S_total_x + S_total_y @ S_total_y + S_total_z @ S_total_z
Sz2_op = S_total_z @ S_total_z

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
rho_S = np.outer(state_S, state_S.conj())
rho_0 = np.kron(rho_S, 0.5 * eye2)

# ==============================================================================
# 3. HAMILTONIAN & MASTER EQUATION GENERATORS
# ==============================================================================
def get_hamiltonian(B0_mag, theta):
    """Constructs the system Hamiltonian with Zeeman and Hyperfine terms."""
    H_static = GAMMA * B0_mag * (np.sin(theta)*(sDx + sAx) + np.cos(theta)*(sDz + sAz))
    H_hyperfine = A_z * (sAz @ Iz)
    return H_static + H_hyperfine

def L_unitary(rho, H_mag):
    """Generator for standard unitary evolution."""
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

# Dictionary routing for cleaner functional calls
GENERATORS = {
    "Haberkorn": L_haberkorn,
    "Jones-Hore": L_hore,
    "Kominis": L_kominis
}

# ==============================================================================
# 4. TIME EVOLUTION & TIME-DOMAIN FISHER CALCULATOR
# ==============================================================================
def get_time_evolution(H, method_name):
    """Calculates rho(t) continuously mapping to the evaluation grid."""
    L_func = GENERATORS[method_name]
    
    def deriv(t, y):
        rho = y.reshape((8, 8))
        d_rho = L_func(rho, H)
        return d_rho.flatten()
        
    y0 = rho_0.flatten()
    sol = solve_ivp(deriv, [0, T_MAX], y0, method='BDF', t_eval=t_eval, rtol=1e-10, atol=1e-12)
    return sol.y  # Shape: (64, N_TIME_STEPS)

def evaluate_fisher_metrics_time(method_name, dtheta=1e-4):
    """Evaluates the QFI, instantaneous speed, and CFI for the given dynamics."""
    H = get_hamiltonian(B0_REF, THETA_FIX)
    H_p = get_hamiltonian(B0_REF, THETA_FIX + dtheta)
    H_m = get_hamiltonian(B0_REF, THETA_FIX - dtheta)
    
    # Extract time series for directional sensitivity
    rho_t   = get_time_evolution(H, method_name)
    rho_p_t = get_time_evolution(H_p, method_name)
    rho_m_t = get_time_evolution(H_m, method_name)
    
    L_func = GENERATORS[method_name]
    
    qfi_arr, v_t_arr = [], []
    cfi_QS_arr, cfi_S2_arr, cfi_Sz2_arr = [], [], []
    
    for k in range(N_TIME_STEPS):
        # Extract and enforce Hermiticity
        rho = 0.5 * (rho_t[:, k].reshape(8,8) + rho_t[:, k].reshape(8,8).conj().T)
        rho_p = 0.5 * (rho_p_t[:, k].reshape(8,8) + rho_p_t[:, k].reshape(8,8).conj().T)
        rho_m = 0.5 * (rho_m_t[:, k].reshape(8,8) + rho_m_t[:, k].reshape(8,8).conj().T)
        
        tr = np.real(np.trace(rho))
        tr_p = np.real(np.trace(rho_p))
        tr_m = np.real(np.trace(rho_m))
        
        # Dead populations yield zero metrics
        if tr < 1e-12 or tr_p < 1e-12 or tr_m < 1e-12:
            qfi_arr.append(0.0)
            v_t_arr.append(0.0)
            cfi_QS_arr.append(0.0)
            cfi_S2_arr.append(0.0)
            cfi_Sz2_arr.append(0.0)
            continue
            
        # Derivatives
        d_rho_theta = (rho_p - rho_m) / (2 * dtheta)
        d_rho_time = L_func(rho, H)
        d_rho_time = 0.5 * (d_rho_time + d_rho_time.conj().T)
        
        # 1. Quantum Fisher Information & Instantaneous Speed
        vals, vecs = np.linalg.eigh(rho)
        vals = np.clip(vals, 0, None)
        
        qfi_theta = 0.0
        qfi_time = 0.0
        
        for i in range(8):
            for j in range(8):
                if vals[i] + vals[j] > 1e-10: 
                    # Theta QFI
                    mat_el_theta = vecs[:, i].conj().T @ d_rho_theta @ vecs[:, j]
                    qfi_theta += 2 * (np.abs(mat_el_theta)**2) / (vals[i] + vals[j])
                    
                    # Time QFI
                    mat_el_time = vecs[:, i].conj().T @ d_rho_time @ vecs[:, j]
                    qfi_time += 2 * (np.abs(mat_el_time)**2) / (vals[i] + vals[j])
                    
        qfi_arr.append(np.real(qfi_theta))
        
        # Calculate speed v(t) = 0.5 * sqrt(QFI_t). Divide by 1e6 for us^-1
        v_t_arr.append((0.5 * np.sqrt(np.real(qfi_time))) / 1e6)
        
        # 2. Classical Fisher Information
        def cfi_of_observable(op):
            exp_O = np.real(np.trace(rho @ op))
            exp_O2 = np.real(np.trace(rho @ op @ op))
            var = exp_O2 - exp_O**2
            d_exp_O = np.real(np.trace(d_rho_theta @ op))
            
            if var > 1e-11:
                return (d_exp_O**2) / var
            return 0.0

        cfi_QS_arr.append(cfi_of_observable(QS))
        cfi_S2_arr.append(cfi_of_observable(S2_op))
        cfi_Sz2_arr.append(cfi_of_observable(Sz2_op))
        
    return qfi_arr, v_t_arr, cfi_QS_arr, cfi_S2_arr, cfi_Sz2_arr

# ==============================================================================
# 5. SIMULATION SWEEP & PLOTTING
# ==============================================================================
def create_time_plot(t_eval_us, y_H, y_JH, y_K, ylabel, filename):
    """Helper to generate metric plots over time."""
    plt.figure(figsize=(10, 6.5))
    plt.plot(t_eval_us, y_H, color='blue', linewidth=2.5, linestyle='--', label='Haberkorn ME')
    plt.plot(t_eval_us, y_JH, color='green', linewidth=2.5, linestyle='-.', label='Jones-Hore ME')
    plt.plot(t_eval_us, y_K, color='red', linewidth=2.5, label='Kominis ME')

    plt.xlabel(r'Time, $t$ ($\mu$s)', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    plt.xlim(0, t_eval_us[-1])
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    
    y_min, y_max = plt.ylim()
    plt.ylim(0, y_max * 1.1)
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

if __name__ == "__main__":
    results = {}
    methods = ["Haberkorn", "Jones-Hore", "Kominis"]
    
    print("Integrating Time Evolution at theta.")
    
    for method in methods:
        print(f"Solving {method} constraints...")
        q, v_t, cQS, cS2, cSz2 = evaluate_fisher_metrics_time(method)
        results[method] = {
            "qfi": q, 
            "v_t": v_t, 
            "cfi_QS": cQS, 
            "cfi_S2": cS2, 
            "cfi_Sz2": cSz2
        }

    
    # Convert seconds to microseconds for plotting
    t_eval_us = t_eval * 1e6 
    
    create_time_plot(t_eval_us, results["Haberkorn"]["qfi"], results["Jones-Hore"]["qfi"], results["Kominis"]["qfi"],
                     ylabel=r'Quantum Fisher Information, QFI', filename="Figure1_Time_QFI.png")
                
    create_time_plot(t_eval_us, results["Haberkorn"]["v_t"], results["Jones-Hore"]["v_t"], results["Kominis"]["v_t"],
                     ylabel=r'Instantaneous speed, $v(t)$ ($\mu$s$^{-1}$)', filename="Figure_Time_Speed_vt.png")
    
    create_time_plot(t_eval_us, results["Haberkorn"]["cfi_QS"], results["Jones-Hore"]["cfi_QS"], results["Kominis"]["cfi_QS"],
                     ylabel=r'CFI (Singlet Yield)', filename="Figure2_Time_CFI_QS.png")
    
    create_time_plot(t_eval_us, results["Haberkorn"]["cfi_S2"], results["Jones-Hore"]["cfi_S2"], results["Kominis"]["cfi_S2"],
                     ylabel=r'CFI ($S^2$)', filename="Figure3_Time_CFI_S2.png")

    create_time_plot(t_eval_us, results["Haberkorn"]["cfi_Sz2"], results["Jones-Hore"]["cfi_Sz2"], results["Kominis"]["cfi_Sz2"],
                     ylabel=r'CFI ($S_z^2$)', filename="Figure4_Time_CFI_Sz2.png")