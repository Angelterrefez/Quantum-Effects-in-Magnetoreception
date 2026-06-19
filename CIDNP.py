"""
CIDNP Radical-Pair Simulation
=================================================================
Extendes the chemically induced dynamic nuclear polarization (CIDNP)
calculations from:

    Tsampourakis & Kominis (2015)
    "Quantum trajectory tests of radical-pair quantum dynamics in CIDNP
     measurements of photosynthetic reaction centers"
    arXiv:1506.08534

The code compares three master-equation formalisms:

  * Haberkorn — the traditional spin-chemistry approach.
  * Jones & Hore — a quantum-measurement-based approach that
    includes random projections the environment does on the ensemble.
  * Kominis — a quantum-measurement-based approach that
    includes real and virtual transitions to vibrational reservoirs of
    the environment.
  

System
------
A radical-ion pair with two unpaired electrons (donor D, acceptor A)
and one nuclear spin (I = 1/2), giving an 8-dimensional Hilbert space.

Basis ordering: |up up up⟩, |up up down⟩, |up down up⟩, |up down down⟩, 
|down up up⟩, |down up down⟩, |down down up⟩, |down down down⟩
(first two indices = electron spins D, A; last = nuclear spin I).

Usage
-----
Run directly::

    python CIDNP.py

Output: PNG plots and CSV files with time traces and integrated yields.

Requirements: numpy, scipy, matplotlib, pandas, tqdm
"""

import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


# =============================================================================
# 1. PHYSICAL CONSTANTS
# =============================================================================

GAMMA_E = 1.7608596e11      # Electron gyromagnetic ratio   [rad s^{-1} T^{-1}]
GAMMA_H = 2.67522e8         # Proton gyromagnetic ratio     [rad s^{-1} T^{-1}]
HBAR    = 1.05457e-34       # Reduced Planck constant       [J s]
KB      = 1.380649e-23      # Boltzmann constant            [J K^{-1}]
MU_B    = 9.2740100657e-24  # Bohr magneton                 [J T^{-1}]
MU_N    = 5.0507832413e-27  # Nuclear magneton              [J T^{-1}]


# =============================================================================
# 2. EXPERIMENTAL PARAMETERS
# =============================================================================
# All frequencies are expressed in rad ns^{-1}, matching the nanosecond timescale
# of radical-pair lifetimes.  Rates k_S, k_T are therefore in ns^{-1}.

B0      = 5.0            # Applied magnetic field              [T]
TEMP    = 233.0          # Temperature                         [K]
DELTA_G = 3.085e-4       # Delta_g between donor and acceptor radicals (for ^{15}N)
G_N     = -0.5664        # Nuclear g-factor (^{15}N)

k_S = 0.05               # Singlet recombination rate  [ns^{-1}] ≈ (20 ns)^{-1}
k_T = 1.0                # Triplet recombination rate  [ns^{-1}] ≈ (1 ns)^{-1}
# Asymmetric rates (k_T ≫ k_S) are representative of photosynthetic reaction
# centres and are the experimentally interesting regime (see paper Sec. IV).

FREQ_SCALE = 1e-9        # Converts to rad ns^{-1}

# Frequency parameters — "double-matching" condition: |Delta_w| = |w_I| = |A|
# This alignment of electron Zeeman splitting, nuclear Zeeman splitting, and
# hyperfine coupling maximises nuclear spin polarisation.
Delta_omega = DELTA_G * MU_B * B0 / HBAR * FREQ_SCALE     # [rad ns^{-1}]
w_I         = G_N   * MU_N * B0 / HBAR * FREQ_SCALE       # [rad ns^{-1}]
A           = -Delta_omega                                # Isotropic HFC [rad ns^{-1}]
B           = A / 2.0                                     # Anisotropic HFC [rad ns^{-1}]

# Thermal nuclear spin polarisation (proton reference, standard in CIDNP).
# CIDNP enhancements are typically 10^3–10^5 times I_thermal.
I_THERMAL = HBAR * GAMMA_H * B0 / (2 * KB * TEMP)       # ~ 2.2 × 10^{-5}

print(f"Delta_ω = {Delta_omega:.4f},  A = {A:.4f},  B = {B:.4f}  [rad ns^{-1}]")
print(f"k_S = {k_S},  k_T = {k_T}  [ns^{-1}]")
print(f"I_thermal = {I_THERMAL:.4e}")

# Simulation time grid
t_max   = 20.0 / k_T    # Integrate until the reaction is essentially complete
n_steps = 10000
dt      = t_max / n_steps
t_eval  = np.linspace(0, t_max, n_steps)

# Monte Carlo sample sizes
N_TRAJ_DIZ = 2000000   # Trajectories for dIz (large N for low noise)
N_TRAJ_TR  = 2000      # Trajectories for Tr{\rho} population decay


# =============================================================================
# 3. SPIN OPERATORS  (8 × 8 Hilbert space)
# =============================================================================

sx = np.array([[0,  1 ],  [1,  0 ]], dtype=complex)
sy = np.array([[0, -1j],  [1j, 0 ]], dtype=complex)
sz = np.array([[1,  0 ],  [0, -1 ]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def kron3(A, B, C):
    """Kronecker product of three 2x2 matrices into the 8x8 space."""
    return np.kron(A, np.kron(B, C))


# Electron spin operators (units of hbar = 1)
s_Dz = 0.5 * kron3(sz, I2, I2)   # Donor electron, z-component
s_Az = 0.5 * kron3(I2, sz, I2)   # Acceptor electron, z-component

# Nuclear spin operators
I_z = 0.5 * kron3(I2, I2, sz)    # Nuclear spin, z-component (the CIDNP observable)
I_x = 0.5 * kron3(I2, I2, sx)    # Nuclear spin, x-component (for anisotropic HFC)

# Singlet and triplet projectors onto the two-electron subspace
s_D_dot_s_A = 0.25 * (kron3(sx, sx, I2) + kron3(sy, sy, I2) + kron3(sz, sz, I2))
Q_S = 0.25 * np.eye(8, dtype=complex) - s_D_dot_s_A
Q_T = np.eye(8, dtype=complex) - Q_S

# Individual triplet-sublevel projectors, used in computing p_coh (Kominis eq.)
_s = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0])     # |S⟩  (2-electron singlet)
_T_p = np.array([1, 0, 0, 0])                          # |T_+⟩
_T_0 = np.array([0, 1/np.sqrt(2),  1/np.sqrt(2), 0])   # |T_0⟩
_T_m = np.array([0, 0, 0, 1])                          # |T_-⟩

T_PROJECTORS = [
    np.kron(np.outer(_T_p, _T_p), I2),
    np.kron(np.outer(_T_0, _T_0), I2),
    np.kron(np.outer(_T_m, _T_m), I2),
]


# =============================================================================
# 4. HAMILTONIAN
# =============================================================================
# H = (Delta_w/2)(s_Az - s_Dz)   [electron chemical shift due to Delta_g]
#   + w_I I_z                    [nuclear Zeeman term]
#   + A s_Az I_z                 [isotropic hyperfine coupling]
#   + B s_Az I_x                 [anisotropic hyperfine coupling]
#
# This form follows eq. (7) of Tsampourakis & Kominis (2015) and the
# parametrisation in Jeschke (JACS 1998).

H = (  (Delta_omega / 2) * s_Az
      - (Delta_omega / 2) * s_Dz
      + w_I * I_z
      + A * (s_Az @ I_z)
      + B * (s_Az @ I_x)  )


# =============================================================================
# 5. INITIAL STATES
# =============================================================================
# The radical pair is created in the singlet electronic state with the nuclear
# spin either spin-up (up) or spin-down (down).  Results are averaged over both
# to simulate an initially unpolarised nuclear spin.

singlet  = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0], dtype=complex)
nuc_up   = np.array([1, 0], dtype=complex)
nuc_down = np.array([0, 1], dtype=complex)

psi_0_up   = np.kron(singlet, nuc_up)
psi_0_down = np.kron(singlet, nuc_down)
rho_0_up   = np.outer(psi_0_up,   psi_0_up.conj())
rho_0_down = np.outer(psi_0_down, psi_0_down.conj())


# =============================================================================
# 6. MASTER EQUATIONS
# =============================================================================

def _pcoh(rho):
    """
    Compute the singlet-triplet coherence measure p_coh (Kritsotakis & Kominis 2014).

    p_coh in [0, 1]:
      0 --> maximally incoherent mixture of S and T states
      1 --> maximally coherent superposition

    Returns
    -------
    p_coh : float
    trace : float  — Tr{rho}, needed by the caller to avoid recomputing it
    """
    trace = np.real(np.trace(rho))
    if trace < 1e-15:
        return 0.0, 0.0

    rho_ST = Q_S @ rho @ Q_T
    rho_TS = Q_T @ rho @ Q_S
    C = sum(np.sqrt(abs(np.trace(rho_ST @ P @ rho_TS))) for P in T_PROJECTORS)

    return float(np.clip((4.0 / 3.0) * C / trace, 0.0, 1.0)), trace


def haberkorn_rhs(t, rho_flat):
    """
    Haberkorn (1976) master equation (eq. 5 in the paper):

      drho/dt = -i[H, rho]
              - (k_S/2)(Q_S rho + rho Q_S)
              - (k_T/2)(Q_T rho + rho Q_T)

    This is the conventional spin-chemistry approach. It can be derived from
    the Kominis equation by setting p_coh = 0 at all times.
    """
    rho = rho_flat.reshape((8, 8))
    drho = (  -1j * (H @ rho - rho @ H)
              - (k_S / 2) * (Q_S @ rho + rho @ Q_S)
              - (k_T / 2) * (Q_T @ rho + rho @ Q_T)  )
    return drho.flatten()


def kominis_rhs(t, rho_flat):
    """
    Kominis master equation (eqs. 1-4 in the paper):

      drho/dt = -i[H, rho]                                    [unitary]
              - (k_S+k_T)/2 · (rho_ST + rho_TS)               [dephasing]
              - (1-p_coh)(k_S rho_SS + k_T rho_TT)            [incoherent reaction]
              - (dr_S+dr_T)/Tr{rho} · (p_coh rho_SS           [coherent reaction]
                                    + p_coh rho_TT
                                    + rho_ST + rho_TS)

    where rho_SS = Q_S rho Q_S, rho_TT = Q_T rho Q_T, etc.
    """
    rho = rho_flat.reshape((8, 8))
    p_coh, trace = _pcoh(rho)
    if trace < 1e-15:
        return np.zeros_like(rho_flat)

    rho_SS = Q_S @ rho @ Q_S
    rho_TT = Q_T @ rho @ Q_T
    rho_ST = Q_S @ rho @ Q_T
    rho_TS = Q_T @ rho @ Q_S

    rate_S = k_S * np.real(np.trace(rho_SS))
    rate_T = k_T * np.real(np.trace(rho_TT))

    drho = (  -1j * (H @ rho - rho @ H)
              - ((k_S + k_T) / 2) * (rho_ST + rho_TS)
              - (1 - p_coh) * (k_S * rho_SS + k_T * rho_TT)
              - ((rate_S + rate_T) / trace)
                * (p_coh * rho_SS + p_coh * rho_TT + rho_ST + rho_TS)  )
    return drho.flatten()


def jones_hore_rhs(t, rho_flat):
    """
    Jones-Hore master equation — extends Haberkorn with additional off-diagonal
    (singlet-triplet) dephasing terms:

      drho/dt = -i[H, rho]
              - (k_S/2)(Q_S rho + rho Q_S)
              - (k_T/2)(Q_T rho + rho Q_T)
              - (k_S+k_T)/2 · (rho_ST + rho_TS)
    """
    rho = rho_flat.reshape((8, 8))
    rho_ST = Q_S @ rho @ Q_T
    rho_TS = Q_T @ rho @ Q_S

    drho = (  -1j * (H @ rho - rho @ H)
              - (k_S / 2) * (Q_S @ rho + rho @ Q_S)
              - (k_T / 2) * (Q_T @ rho + rho @ Q_T)
              - ((k_S + k_T) / 2) * (rho_ST + rho_TS)  )
    return drho.flatten()


# =============================================================================
# 7. OBSERVABLES FROM MASTER-EQUATION SOLUTIONS
# =============================================================================

def calc_dIz_ME(sol_y):
    """
    Nuclear spin polarisation deposited to reaction products per time step
    (master-equation version, eq. 9 in the paper):

        dI_z = dt · [k_S Tr{I_z Q_S rho Q_S} + k_T Tr{I_z Q_T rho Q_T}]

    Normalised by I_thermal.

    Parameters
    ----------
    sol_y : ndarray, shape (64, n_steps)
        Flattened density matrix at each time step (output of solve_ivp).

    Returns
    -------
    ndarray, shape (n_steps,)
    """
    dIz = np.array([
        np.real(k_S * np.trace(I_z @ Q_S @ rho_flat.reshape(8, 8) @ Q_S)
              + k_T * np.trace(I_z @ Q_T @ rho_flat.reshape(8, 8) @ Q_T))
        for rho_flat in sol_y.T
    ])
    return (dIz * dt) / I_THERMAL


def calc_trace_ME(sol_y):
    """
    Radical-pair survival probability Tr{rho(t)} from the master equation.

    Parameters
    ----------
    sol_y : ndarray, shape (64, n_steps)

    Returns
    -------
    ndarray, shape (n_steps,)
    """
    return np.array([
        np.real(np.trace(rho_flat.reshape(8, 8)))
        for rho_flat in sol_y.T
    ])


# =============================================================================
# 8. VECTORISED MONTE CARLO
# =============================================================================

def run_monte_carlo(method, psi_init, t_grid, dt, N, compute_dIz=True, desc="MC"):
    """
    Simulate N quantum trajectories simultaneously using NumPy broadcasting.

    At each time step every active trajectory independently draws a single
    uniform random number and is assigned to one of several event channels
    according to the instantaneous probabilities (Tables I and II in the paper).

    Parameters
    ----------
    method : {"Kominis", "Haberkorn", "Jones-Hore"}
        Selects the event table and probability expressions.
    psi_init : ndarray, shape (8,)
        Initial pure state.
    t_grid : ndarray, shape (n_steps,)
        Time points.
    dt : float
        Time step.
    N : int
        Number of trajectories.
    compute_dIz : bool
        If True, accumulate nuclear spin dIz at recombination events.
        If False, count surviving trajectories at each time step (--> Tr{rho}).
    desc : str
        Label shown in the tqdm progress bar.

    Returns
    -------
    result : ndarray, shape (n_steps,)
        If compute_dIz: dI_z(t) / I_thermal.
        If not compute_dIz: fraction of surviving trajectories.
    """
    # Initialise: columns are individual trajectories
    psi    = np.outer(psi_init, np.ones(N, dtype=complex))   # (8, N)
    active = np.ones(N, dtype=bool)

    result = np.zeros(len(t_grid))
    if not compute_dIz:
        result[0] = N   # All trajectories alive at t=0

    # Pre-compute the unitary propagator for a single time step
    U = la.expm(-1j * H * dt)

    for i in tqdm(range(1, len(t_grid)), desc=desc, ncols=80, leave=False):
        idx = np.where(active)[0]
        if len(idx) == 0:
            break

        psi_t = psi[:, idx]          # (8, n_active)
        n_act = len(idx)

        # Singlet and triplet probabilities for each active trajectory
        P_S = np.sum(psi_t.conj() * (Q_S @ psi_t), axis=0).real   # (n_active,)
        P_T = np.sum(psi_t.conj() * (Q_T @ psi_t), axis=0).real

        r = np.random.rand(n_act)   # One random number per trajectory

        # ------------------------------------------------------------------ #
        # Event probabilities and masks (partition of [0, 1])
        # ------------------------------------------------------------------ #
        if method == "Kominis":
            # Five events (Table I in the paper):
            #   K1: singlet-subspace projection (dephasing), prob = (k_S+k_T)/2 dt P_S
            #   K2: triplet-subspace projection (dephasing), prob = (k_S+k_T)/2 dt P_T
            #   K3: singlet recombination,                   prob = k_S dt P_S
            #   K4: triplet recombination,                   prob = k_T dt P_T
            #   K5: coherent (Hamiltonian) evolution,        prob = remainder
            dp_S = 0.5 * (k_S + k_T) * dt * P_S
            dp_T = 0.5 * (k_S + k_T) * dt * P_T
            dr_S = k_S * dt * P_S
            dr_T = k_T * dt * P_T

            c1 = dp_S
            c2 = c1 + dp_T
            c3 = c2 + dr_S
            c4 = c3 + dr_T

            m_K1  = r < c1
            m_K2  = (r >= c1) & (r < c2)
            m_K3  = (r >= c2) & (r < c3)
            m_K4  = (r >= c3) & (r < c4)
            m_K5  = r >= c4

            # K1 — project onto singlet subspace and renormalise
            if np.any(m_K1):
                sub = np.where(m_K1)[0]
                pn  = Q_S @ psi_t[:, sub]
                psi[:, idx[sub]] = pn / np.linalg.norm(pn, axis=0)

            # K2 — project onto triplet subspace and renormalise
            if np.any(m_K2):
                sub = np.where(m_K2)[0]
                pn  = Q_T @ psi_t[:, sub]
                psi[:, idx[sub]] = pn / np.linalg.norm(pn, axis=0)

            # K3 — singlet recombination: record nuclear spin, deactivate
            if np.any(m_K3):
                sub = np.where(m_K3)[0]
                if compute_dIz:
                    pn  = Q_S @ psi_t[:, sub]
                    pn /= np.linalg.norm(pn, axis=0)
                    result[i] += np.sum(np.sum(pn.conj() * (I_z @ pn), axis=0).real)
                active[idx[sub]] = False

            # K4 — triplet recombination: record nuclear spin, deactivate
            if np.any(m_K4):
                sub = np.where(m_K4)[0]
                if compute_dIz:
                    pn  = Q_T @ psi_t[:, sub]
                    pn /= np.linalg.norm(pn, axis=0)
                    result[i] += np.sum(np.sum(pn.conj() * (I_z @ pn), axis=0).real)
                active[idx[sub]] = False

            # K5 — coherent evolution
            if np.any(m_K5):
                sub = np.where(m_K5)[0]
                pn  = U @ psi_t[:, sub]
                psi[:, idx[sub]] = pn / np.linalg.norm(pn, axis=0)

        elif method == "Jones-Hore":
            # Five events analogous to Kominis but with swapped dephasing rates:
            #   J1: singlet projection, prob = k_T dt P_S
            #   J2: triplet projection, prob = k_S dt P_T
            #   J3/J4/J5: same as K3–K5
            dp_S = k_T * dt * P_S
            dp_T = k_S * dt * P_T
            dr_S = k_S * dt * P_S
            dr_T = k_T * dt * P_T

            c1 = dp_S
            c2 = c1 + dp_T
            c3 = c2 + dr_S
            c4 = c3 + dr_T

            m_J1 = r < c1
            m_J2 = (r >= c1) & (r < c2)
            m_J3 = (r >= c2) & (r < c3)
            m_J4 = (r >= c3) & (r < c4)
            m_J5 = r >= c4

            if np.any(m_J1):
                sub = np.where(m_J1)[0]
                pn  = Q_S @ psi_t[:, sub]
                psi[:, idx[sub]] = pn / np.linalg.norm(pn, axis=0)

            if np.any(m_J2):
                sub = np.where(m_J2)[0]
                pn  = Q_T @ psi_t[:, sub]
                psi[:, idx[sub]] = pn / np.linalg.norm(pn, axis=0)

            if np.any(m_J3):
                sub = np.where(m_J3)[0]
                if compute_dIz:
                    pn  = Q_S @ psi_t[:, sub]
                    pn /= np.linalg.norm(pn, axis=0)
                    result[i] += np.sum(np.sum(pn.conj() * (I_z @ pn), axis=0).real)
                active[idx[sub]] = False

            if np.any(m_J4):
                sub = np.where(m_J4)[0]
                if compute_dIz:
                    pn  = Q_T @ psi_t[:, sub]
                    pn /= np.linalg.norm(pn, axis=0)
                    result[i] += np.sum(np.sum(pn.conj() * (I_z @ pn), axis=0).real)
                active[idx[sub]] = False

            if np.any(m_J5):
                sub = np.where(m_J5)[0]
                pn  = U @ psi_t[:, sub]
                psi[:, idx[sub]] = pn / np.linalg.norm(pn, axis=0)

        elif method == "Haberkorn":
            # Three events (Table II in the paper):
            #   H1: singlet recombination, prob = k_S dt P_S
            #   H2: triplet recombination, prob = k_T dt P_T
            #   H3: coherent evolution,    prob = remainder
            # No dephasing jumps — surviving pairs evolve unitarily.
            dr_S = k_S * dt * P_S
            dr_T = k_T * dt * P_T

            c1 = dr_S
            c2 = c1 + dr_T

            m_H1 = r < c1
            m_H2 = (r >= c1) & (r < c2)
            m_H3 = r >= c2

            if np.any(m_H1):
                sub = np.where(m_H1)[0]
                if compute_dIz:
                    pn  = Q_S @ psi_t[:, sub]
                    pn /= np.linalg.norm(pn, axis=0)
                    result[i] += np.sum(np.sum(pn.conj() * (I_z @ pn), axis=0).real)
                active[idx[sub]] = False

            if np.any(m_H2):
                sub = np.where(m_H2)[0]
                if compute_dIz:
                    pn  = Q_T @ psi_t[:, sub]
                    pn /= np.linalg.norm(pn, axis=0)
                    result[i] += np.sum(np.sum(pn.conj() * (I_z @ pn), axis=0).real)
                active[idx[sub]] = False

            if np.any(m_H3):
                sub = np.where(m_H3)[0]
                pn  = U @ psi_t[:, sub]
                psi[:, idx[sub]] = pn / np.linalg.norm(pn, axis=0)

        else:
            raise ValueError(f"Unknown method '{method}'. "
                             "Choose 'Kominis', 'Haberkorn', or 'Jones-Hore'.")

        if not compute_dIz:
            result[i] = np.sum(active)

    if compute_dIz:
        return (result / N) / I_THERMAL
    else:
        return result / N


# =============================================================================
# 9. PLOTTING HELPER
# =============================================================================

def plot_and_save(mc_data, me_data, title, ylabel, filename, is_diz=True, integrals=None):
    """
    Create a single comparison plot (Monte Carlo vs master equation) and save it.

    Parameters
    ----------
    mc_data, me_data : ndarray, shape (n_steps,)
        Monte Carlo and master-equation traces.
    title : str
    ylabel : str
    filename : str
    is_diz : bool
        True for dI_z plots; False for Tr{rho} plots (different line styles).
    integrals : tuple (float, float) or None
        (master-equation integral, Monte Carlo integral) shown as an inset table.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    time_axis = t_eval * k_T   # Express time in units of 1/k_T

    if is_diz:
        ax.plot(time_axis, mc_data, color='silver', lw=0.8, alpha=0.9, label='Monte Carlo')
        ax.plot(time_axis, me_data, color='black',  lw=2.0,             label='Master Equation')
    else:
        ax.plot(time_axis, mc_data, color='silver', lw=2.0,             label='Monte Carlo')
        ax.plot(time_axis, me_data, color='black',  lw=1.5, ls='--',   label='Master Equation')
        ax.set_ylim(-0.05, 1.05)

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel(r'time  (units of $1/k_T$)')
    ax.set_ylabel(ylabel)
    ax.set_xlim(-1, 21)
    ax.legend()

    if integrals is not None:
        text = (f"Integrated yield\n"
                f"Master eq.:  {integrals[0]:.0f}\n"
                f"Monte Carlo: {integrals[1]:.0f}")
        ax.text(0.95, 0.05, text, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9, ec='gray'))

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


# =============================================================================
# 10. MAIN
# =============================================================================

if __name__ == '__main__':

    # ------------------------------------------------------------------
    # Master equations
    # ------------------------------------------------------------------
    print("\n=== Master equations ===")
    solve_kw = dict(t_eval=t_eval, max_step=dt)

    sol_H_dn = solve_ivp(haberkorn_rhs,  [0, t_max], rho_0_down.flatten(), **solve_kw)
    sol_H_up = solve_ivp(haberkorn_rhs,  [0, t_max], rho_0_up.flatten(),   **solve_kw)
    sol_K_dn = solve_ivp(kominis_rhs,    [0, t_max], rho_0_down.flatten(), **solve_kw)
    sol_K_up = solve_ivp(kominis_rhs,    [0, t_max], rho_0_up.flatten(),   **solve_kw)
    sol_J_dn = solve_ivp(jones_hore_rhs, [0, t_max], rho_0_down.flatten(), **solve_kw)
    sol_J_up = solve_ivp(jones_hore_rhs, [0, t_max], rho_0_up.flatten(),   **solve_kw)

    # Average over spin-up and spin-down initial conditions (unpolarised nucleus)
    ME_H_diz = (calc_dIz_ME(sol_H_dn.y) + calc_dIz_ME(sol_H_up.y)) / 2
    ME_K_diz = (calc_dIz_ME(sol_K_dn.y) + calc_dIz_ME(sol_K_up.y)) / 2
    ME_J_diz = (calc_dIz_ME(sol_J_dn.y) + calc_dIz_ME(sol_J_up.y)) / 2

    ME_H_tr = (calc_trace_ME(sol_H_dn.y) + calc_trace_ME(sol_H_up.y)) / 2
    ME_K_tr = (calc_trace_ME(sol_K_dn.y) + calc_trace_ME(sol_K_up.y)) / 2
    ME_J_tr = (calc_trace_ME(sol_J_dn.y) + calc_trace_ME(sol_J_up.y)) / 2

    # ------------------------------------------------------------------
    # Monte Carlo for dIz
    # ------------------------------------------------------------------
    print(f"\n=== Monte Carlo — dIz  (N = {N_TRAJ_DIZ:,}) ===")
    MC_H_diz = (run_monte_carlo("Haberkorn",  psi_0_down, t_eval, dt, N_TRAJ_DIZ, True,  "H dIz ↓") +
                run_monte_carlo("Haberkorn",  psi_0_up,   t_eval, dt, N_TRAJ_DIZ, True,  "H dIz ↑")) / 2
    MC_K_diz = (run_monte_carlo("Kominis",    psi_0_down, t_eval, dt, N_TRAJ_DIZ, True,  "K dIz ↓") +
                run_monte_carlo("Kominis",    psi_0_up,   t_eval, dt, N_TRAJ_DIZ, True,  "K dIz ↑")) / 2
    MC_J_diz = (run_monte_carlo("Jones-Hore", psi_0_down, t_eval, dt, N_TRAJ_DIZ, True,  "J dIz ↓") +
                run_monte_carlo("Jones-Hore", psi_0_up,   t_eval, dt, N_TRAJ_DIZ, True,  "J dIz ↑")) / 2

    # ------------------------------------------------------------------
    # Monte Carlo for Tr{rho}
    # ------------------------------------------------------------------
    print(f"\n=== Monte Carlo — Tr{{rho}}  (N = {N_TRAJ_TR:,}) ===")
    MC_H_tr = (run_monte_carlo("Haberkorn",  psi_0_down, t_eval, dt, N_TRAJ_TR, False, "H Tr ↓") +
               run_monte_carlo("Haberkorn",  psi_0_up,   t_eval, dt, N_TRAJ_TR, False, "H Tr ↑")) / 2
    MC_K_tr = (run_monte_carlo("Kominis",    psi_0_down, t_eval, dt, N_TRAJ_TR, False, "K Tr ↓") +
               run_monte_carlo("Kominis",    psi_0_up,   t_eval, dt, N_TRAJ_TR, False, "K Tr ↑")) / 2
    MC_J_tr = (run_monte_carlo("Jones-Hore", psi_0_down, t_eval, dt, N_TRAJ_TR, False, "J Tr ↓") +
               run_monte_carlo("Jones-Hore", psi_0_up,   t_eval, dt, N_TRAJ_TR, False, "J Tr ↑")) / 2

    # ------------------------------------------------------------------
    # Integrated yields  \int dIz dt  (the key number reported in Table III)
    # ------------------------------------------------------------------
    int_K_ME, int_K_MC = np.sum(ME_K_diz), np.sum(MC_K_diz)
    int_H_ME, int_H_MC = np.sum(ME_H_diz), np.sum(MC_H_diz)
    int_J_ME, int_J_MC = np.sum(ME_J_diz), np.sum(MC_J_diz)

    print("\n=== Integrated \int dIz/I_thermal dt ===")
    print(f"{'':20s} {'Master Eq.':>15s} {'Monte Carlo':>15s}")
    print(f"{'Kominis':20s} {int_K_ME:>15.0f} {int_K_MC:>15.0f}")
    print(f"{'Haberkorn':20s} {int_H_ME:>15.0f} {int_H_MC:>15.0f}")
    print(f"{'Jones-Hore':20s} {int_J_ME:>15.0f} {int_J_MC:>15.0f}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\n=== Saving plots ===")
    plot_and_save(MC_K_diz, ME_K_diz,
                  "Kominis — nuclear spin polarisation",
                  r"$dI_z / I_\mathrm{thermal}$",
                  "kominis_dIz.png", is_diz=True,
                  integrals=(int_K_ME, int_K_MC))

    plot_and_save(MC_K_tr, ME_K_tr,
                  "Kominis — radical-pair survival",
                  r"$\mathrm{Tr}\{\hat{\rho}\}$",
                  "kominis_trace.png", is_diz=False)

    plot_and_save(MC_H_diz, ME_H_diz,
                  "Haberkorn — nuclear spin polarisation",
                  r"$dI_z / I_\mathrm{thermal}$",
                  "haberkorn_dIz.png", is_diz=True,
                  integrals=(int_H_ME, int_H_MC))

    plot_and_save(MC_H_tr, ME_H_tr,
                  "Haberkorn — radical-pair survival",
                  r"$\mathrm{Tr}\{\hat{\rho}\}$",
                  "haberkorn_trace.png", is_diz=False)

    plot_and_save(MC_J_diz, ME_J_diz,
                  "Jones-Hore — nuclear spin polarisation",
                  r"$dI_z / I_\mathrm{thermal}$",
                  "jones_hore_dIz.png", is_diz=True,
                  integrals=(int_J_ME, int_J_MC))

    plot_and_save(MC_J_tr, ME_J_tr,
                  "Jones-Hore — radical-pair survival",
                  r"$\mathrm{Tr}\{\hat{\rho}\}$",
                  "jones_hore_trace.png", is_diz=False)

    # ------------------------------------------------------------------
    # Save data to CSV
    # ------------------------------------------------------------------
    print("\n=== Saving CSV data ===")
    time_axis = t_eval * k_T

    pd.DataFrame({
        'time_kT_units': time_axis,
        'Kominis_ME_dIz':    ME_K_diz,
        'Kominis_MC_dIz':    MC_K_diz,
        'Kominis_ME_TrRho':  ME_K_tr,
        'Kominis_MC_TrRho':  MC_K_tr,
        'Haberkorn_ME_dIz':  ME_H_diz,
        'Haberkorn_MC_dIz':  MC_H_diz,
        'Haberkorn_ME_TrRho':ME_H_tr,
        'Haberkorn_MC_TrRho':MC_H_tr,
        'JonesHore_ME_dIz':  ME_J_diz,
        'JonesHore_MC_dIz':  MC_J_diz,
        'JonesHore_ME_TrRho':ME_J_tr,
        'JonesHore_MC_TrRho':MC_J_tr,
    }).to_csv('simulation_traces.csv', index=False)

    pd.DataFrame({
        'Formalism':           ['Kominis', 'Haberkorn', 'Jones-Hore'],
        'ME_integral_dIz':     [int_K_ME,  int_H_ME,   int_J_ME],
        'MC_integral_dIz':     [int_K_MC,  int_H_MC,   int_J_MC],
    }).to_csv('simulation_integrals.csv', index=False)

    print("Done.")