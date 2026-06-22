"""
Singlet-Triplet Coherence Dynamics: Two-Spin vs Guo Hamiltonian
=================================================================
This script computes and plots the time evolution of singlet-triplet (S-T)
coherence for a radical pair under three master-equation formalisms:

  * Haberkorn.
  * Jones & Hore .
  * Kominis.

Two physical settings are compared side by side:

  Figure 1 -- Two-spin model (no nuclear spin, 4x4 Hilbert space).
              A simple Zeeman Hamiltonian with different Larmor frequencies
              on donor and acceptor drives S-T0 mixing. This is a clean
              pedagogical case with no hyperfine coupling.

  Figure 2 -- Full Hamiltonian from Guo et al. (2017), Sci. Rep. 7:5826
              (8x8 Hilbert space, one nuclear spin on the acceptor).
              Parameters match the avian magnetoreception model: B0 = 46 uT,
              A_z = 6*gamma*B0, theta = pi/4, kS = 1e4 s^-1, kT = 1e6 s^-1.
              Time is zoomed to the first 5 us to resolve the coherent
              oscillations driven by the hyperfine coupling before the fast
              triplet decay wipes them out.

Two coherence measures are tracked at each time step:

  |C_total|    = sqrt( Tr{ Q_S rho Q_T * Q_T rho Q_S } )
                 Total S-T coherence in the density matrix.

  |C_{S-T0}|   = sqrt( Tr{ Q_S rho Q_T * P_{T0} * Q_T rho Q_S } )
                 Projection onto the S-T0 sector, the dominant mixing channel
                 in an axially symmetric system.

Usage
-----
    python ST_coherence.py

Output: ST_coherence_twospins.png, ST_coherence_guo.png

Requirements: numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# =============================================================================
# 1. PAULI MATRICES
# =============================================================================

sx = np.array([[0,  1 ], [1,  0 ]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0 ]], dtype=complex)
sz = np.array([[1,  0 ], [0, -1 ]], dtype=complex)
I2 = np.eye(2, dtype=complex)


# =============================================================================
# 2. TWO-SPIN MODEL  (4x4 Hilbert space, no nuclear spin)
# =============================================================================
# Basis ordering: |uu>, |ud>, |du>, |dd>  (donor x acceptor electrons).
#
# The Hamiltonian is a pure Zeeman term with different Larmor frequencies on
# each radical, which drives coherent S-T0 mixing. This is the minimal model
# to observe S-T oscillations with no hyperfine coupling complication.

# --- Parameters ---
kS_2 = 0.05    # Singlet recombination rate [ns^-1]  (1/kT sets the time unit)
kT_2 = 1.0     # Triplet recombination rate [ns^-1]
OMEGA_D = 1.5  # Donor Larmor frequency  [rad ns^-1]
OMEGA_A = 0.5  # Acceptor Larmor frequency [rad ns^-1]

T_MAX_2 = 10.0 / kT_2
N_STEPS_2 = 2000
t_eval_2 = np.linspace(0, T_MAX_2, N_STEPS_2)

# --- Spin operators (4x4) ---
sDz_2 = 0.5 * np.kron(sz, I2)
sAz_2 = 0.5 * np.kron(I2, sz)
sDx_2 = 0.5 * np.kron(sx, I2)
sAx_2 = 0.5 * np.kron(I2, sx)
sDy_2 = 0.5 * np.kron(sy, I2)
sAy_2 = 0.5 * np.kron(I2, sy)

sD_dot_sA_2 = sDx_2 @ sAx_2 + sDy_2 @ sAy_2 + sDz_2 @ sAz_2
QS_2 = 0.25 * np.eye(4, dtype=complex) - sD_dot_sA_2
QT_2 = np.eye(4, dtype=complex) - QS_2

# Triplet sublevel projectors (4x4)
_Tp = np.array([1, 0, 0, 0])
_T0 = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0])
_Tm = np.array([0, 0, 0, 1])
T_PROJ_2 = [
    np.outer(_Tp, _Tp),
    np.outer(_T0, _T0),
    np.outer(_Tm, _Tm),
]

# Initial state: singlet electron state
rho_0_2 = QS_2 / np.trace(QS_2)

# Hamiltonian: different Zeeman frequencies on donor and acceptor
H_2 = OMEGA_D * sDz_2 + OMEGA_A * sAz_2


# =============================================================================
# 3. GUO ET AL. MODEL  (8x8 Hilbert space, one nuclear spin)
# =============================================================================
# Basis ordering: |uu u>, |uu d>, |ud u>, |ud d>, |du u>, |du d>, |dd u>, |dd d>
# (first two indices = electron spins D, A; last = nuclear spin I).
#
# The Hamiltonian follows Guo et al. (2017) eq. (1): isotropic electron Zeeman
# plus an axially symmetric hyperfine coupling on the acceptor only.
# All frequencies are expressed in us^-1 (microseconds) to keep the ODE
# solver well-conditioned: the Larmor frequency (~4 us^-1) and the
# recombination rates (0.01 and 1 us^-1) are then of comparable magnitude,
# avoiding the stiffness that arises when working in SI units directly.

# --- Parameters (Guo et al. 2017) ---
GAMMA  = 8.8e4    # Electron gyromagnetic ratio [rad s^-1 uT^-1]
B0     = 46.0     # Geomagnetic field strength [uT]
A_Z_SI = 6 * GAMMA * B0    # Axial HFC in SI [rad s^-1]
kS_SI  = 1e4      # Singlet recombination rate [s^-1]
kT_SI  = 1e6      # Triplet recombination rate [s^-1]
THETA  = np.pi / 4

# Convert to us^-1
US = 1e-6
kS_8 = kS_SI * US
kT_8 = kT_SI * US
w0   = GAMMA * B0 * US
Az   = A_Z_SI * US

# Time grid: zoom in on the first 5 us to resolve the fast coherent oscillations
# (period ~ 2*pi/w0 ~ 1.5 us) before the triplet channel has fully decayed
T_MAX_8 = 5.0     # [us]
N_STEPS_8 = 5000
t_eval_8 = np.linspace(0, T_MAX_8, N_STEPS_8)

# --- Spin operators (8x8) ---
def kron3(A, B, C):
    return np.kron(A, np.kron(B, C))

sDx_8 = 0.5 * kron3(sx, I2, I2);  sDy_8 = 0.5 * kron3(sy, I2, I2);  sDz_8 = 0.5 * kron3(sz, I2, I2)
sAx_8 = 0.5 * kron3(I2, sx, I2);  sAy_8 = 0.5 * kron3(I2, sy, I2);  sAz_8 = 0.5 * kron3(I2, sz, I2)
Iz_8  = 0.5 * kron3(I2, I2, sz)

sD_dot_sA_8 = sDx_8 @ sAx_8 + sDy_8 @ sAy_8 + sDz_8 @ sAz_8
QS_8 = 0.25 * np.eye(8, dtype=complex) - sD_dot_sA_8
QT_8 = np.eye(8, dtype=complex) - QS_8

# Triplet sublevel projectors (8x8)
T_PROJ_8 = [
    np.kron(np.outer(_Tp, _Tp), I2),
    np.kron(np.outer(_T0, _T0), I2),
    np.kron(np.outer(_Tm, _Tm), I2),
]

# Initial state: singlet electrons, nuclear spin completely mixed
state_S = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0], dtype=complex)
rho_S   = np.outer(state_S, state_S.conj())
rho_0_8 = np.kron(rho_S, 0.5 * I2)

# Hamiltonian (eq. 1 of Guo et al., scaled to us^-1)
H_8 = (  w0 * (np.sin(THETA) * (sDx_8 + sAx_8) + np.cos(THETA) * (sDz_8 + sAz_8))
        + Az * (sAz_8 @ Iz_8)  )


# =============================================================================
# 4. MASTER EQUATIONS
# =============================================================================
# The three Liouvillians are implemented as factory functions that close over
# the model-specific operators (QS, QT, T_PROJ, kS, kT) so the same logic
# works for both the 4x4 and 8x8 cases without code duplication.

def make_liouvillians(QS, QT, T_proj, kS, kT, H):
    """
    Return the three right-hand-side superoperators for solve_ivp, parameterised
    by the projectors, rates, and Hamiltonian of a given model.

    All three return the flattened derivative of rho as expected by solve_ivp.
    """
    dim = QS.shape[0]

    def _pcoh(rho):
        tr = np.real(np.trace(rho))
        if tr < 1e-15:
            return 0.0, 0.0
        rho_ST = QS @ rho @ QT
        rho_TS = QT @ rho @ QS
        C = sum(np.sqrt(abs(np.trace(rho_ST @ P @ rho_TS))) for P in T_proj)
        return float(np.clip((4.0 / 3.0) * C / tr, 0.0, 1.0)), tr

    def haberkorn(t, y):
        rho = y.reshape((dim, dim))
        drho = (  -1j * (H @ rho - rho @ H)
                  - (kS / 2) * (QS @ rho + rho @ QS)
                  - (kT / 2) * (QT @ rho + rho @ QT)  )
        return drho.flatten()

    def jones_hore(t, y):
        rho = y.reshape((dim, dim))
        rho_ST = QS @ rho @ QT
        rho_TS = QT @ rho @ QS
        drho = (  -1j * (H @ rho - rho @ H)
                  - (kS / 2) * (QS @ rho + rho @ QS)
                  - (kT / 2) * (QT @ rho + rho @ QT)
                  - ((kS + kT) / 2) * (rho_ST + rho_TS)  )
        return drho.flatten()

    def kominis(t, y):
        rho = y.reshape((dim, dim))
        p_coh, tr = _pcoh(rho)
        if tr < 1e-15:
            return np.zeros(dim * dim)
        rho_SS = QS @ rho @ QS
        rho_TT = QT @ rho @ QT
        rho_ST = QS @ rho @ QT
        rho_TS = QT @ rho @ QS
        rate_S = kS * np.real(np.trace(rho_SS))
        rate_T = kT * np.real(np.trace(rho_TT))
        drho = (  -1j * (H @ rho - rho @ H)
                  - ((kS + kT) / 2) * (rho_ST + rho_TS)
                  - (1 - p_coh) * (kS * rho_SS + kT * rho_TT)
                  - ((rate_S + rate_T) / tr)
                    * (p_coh * rho_SS + p_coh * rho_TT + rho_ST + rho_TS)  )
        return drho.flatten()

    return haberkorn, jones_hore, kominis


# =============================================================================
# 5. COHERENCE EXTRACTION
# =============================================================================

def extract_coherences(sol_y, QS, QT, P_T0):
    """
    Compute |C_total|(t) and |C_{S-T0}|(t) from a solve_ivp solution array.

    |C_total|  = sqrt( Tr{ rho_ST * rho_TS } )
                 where rho_ST = Q_S rho Q_T and rho_TS = Q_T rho Q_S.

    |C_{S-T0}| = sqrt( Tr{ rho_ST * P_{T0} * rho_TS } )
                 The T0 projection captures the dominant mixing channel in
                 systems with axial symmetry.

    Parameters
    ----------
    sol_y : ndarray, shape (dim^2, n_steps) -- flattened rho at each time step
    QS, QT : ndarray -- singlet/triplet projectors
    P_T0   : ndarray -- T0 sublevel projector

    Returns
    -------
    C_total : ndarray, shape (n_steps,)
    C_ST0   : ndarray, shape (n_steps,)
    """
    dim = QS.shape[0]
    C_total, C_ST0 = [], []
    for y in sol_y.T:
        rho    = y.reshape((dim, dim))
        rho_ST = QS @ rho @ QT
        rho_TS = QT @ rho @ QS
        C_total.append(np.sqrt(abs(np.trace(rho_ST @ rho_TS))))
        C_ST0.append(np.sqrt(abs(np.trace(rho_ST @ P_T0 @ rho_TS))))
    return np.array(C_total), np.array(C_ST0)


# =============================================================================
# 6. RUN BOTH MODELS
# =============================================================================

print("=== Two-spin model (4x4) ===")
hab_2, jh_2, kom_2 = make_liouvillians(QS_2, QT_2, T_PROJ_2, kS_2, kT_2, H_2)

sol_hab_2 = solve_ivp(hab_2, [0, T_MAX_2], rho_0_2.flatten(), t_eval=t_eval_2, method='RK45')
sol_jh_2  = solve_ivp(jh_2,  [0, T_MAX_2], rho_0_2.flatten(), t_eval=t_eval_2, method='RK45')
sol_kom_2 = solve_ivp(kom_2, [0, T_MAX_2], rho_0_2.flatten(), t_eval=t_eval_2, method='RK45')

P_T0_2 = T_PROJ_2[1]   # middle entry is T0
C_tot_hab_2, C_st0_hab_2 = extract_coherences(sol_hab_2.y, QS_2, QT_2, P_T0_2)
C_tot_jh_2,  C_st0_jh_2  = extract_coherences(sol_jh_2.y,  QS_2, QT_2, P_T0_2)
C_tot_kom_2, C_st0_kom_2 = extract_coherences(sol_kom_2.y, QS_2, QT_2, P_T0_2)
print("  done.")

print("=== Guo et al. model (8x8) ===")
hab_8, jh_8, kom_8 = make_liouvillians(QS_8, QT_8, T_PROJ_8, kS_8, kT_8, H_8)

sol_hab_8 = solve_ivp(hab_8, [0, T_MAX_8], rho_0_8.flatten(), t_eval=t_eval_8, method='RK45')
sol_jh_8  = solve_ivp(jh_8,  [0, T_MAX_8], rho_0_8.flatten(), t_eval=t_eval_8, method='RK45')
sol_kom_8 = solve_ivp(kom_8, [0, T_MAX_8], rho_0_8.flatten(), t_eval=t_eval_8, method='RK45')

P_T0_8 = T_PROJ_8[1]
C_tot_hab_8, C_st0_hab_8 = extract_coherences(sol_hab_8.y, QS_8, QT_8, P_T0_8)
C_tot_jh_8,  C_st0_jh_8  = extract_coherences(sol_jh_8.y,  QS_8, QT_8, P_T0_8)
C_tot_kom_8, C_st0_kom_8 = extract_coherences(sol_kom_8.y, QS_8, QT_8, P_T0_8)
print("  done.")


# =============================================================================
# 7. PLOTTING
# =============================================================================

def plot_coherences(t, data, xlabel, ylim, filename):
    """
    Plot total and S-T0 coherence for the three formalisms.

    Each formalism gets a distinct colour (blue = Haberkorn, green = Jones-Hore,
    red = Kominis). Within each colour the solid thick line is |C_total| and
    the dashed thin line is |C_{S-T0}|.

    Parameters
    ----------
    t        : ndarray -- time axis
    data     : dict mapping formalism name to (C_total, C_ST0) tuple
    xlabel   : str -- x-axis label
    ylim     : tuple (ymin, ymax)
    filename : str -- output file path
    """
    colours = {"Haberkorn": "blue", "Jones-Hore": "green", "Kominis": "red"}
    styles  = {"total": ("-",  3, 0.5), "st0": ("--", 2, 1.0)}

    fig, ax = plt.subplots(figsize=(6, 5))

    for name, (C_tot, C_st0) in data.items():
        c = colours[name]
        ls_tot, lw_tot, al_tot = styles["total"]
        ls_st0, lw_st0, al_st0 = styles["st0"]
        ax.plot(t, C_tot, color=c, ls=ls_tot, lw=lw_tot, alpha=al_tot,
                label=rf'$|C_\mathrm{{total}}|$ {name}')
        ax.plot(t, C_st0, color=c, ls=ls_st0, lw=lw_st0, alpha=al_st0,
                label=rf'$|C_{{S\text{{-}}T_0}}|$ {name}')

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('Coherence magnitude', fontsize=14)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=12)

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


plot_coherences(
    t_eval_2,
    {
        "Haberkorn":  (C_tot_hab_2, C_st0_hab_2),
        "Jones-Hore": (C_tot_jh_2,  C_st0_jh_2),
        "Kominis":    (C_tot_kom_2, C_st0_kom_2),
    },
    xlabel=r'Time (units of $1/k_T$)',
    ylim=(0, 0.40),
    filename="ST_coherence_twospins.png",
)

plot_coherences(
    t_eval_8,
    {
        "Haberkorn":  (C_tot_hab_8, C_st0_hab_8),
        "Jones-Hore": (C_tot_jh_8,  C_st0_jh_8),
        "Kominis":    (C_tot_kom_8, C_st0_kom_8),
    },
    xlabel=r'Time ($\mu$s)',
    ylim=(0, 0.40),
    filename="ST_coherence_guo.png",
)
