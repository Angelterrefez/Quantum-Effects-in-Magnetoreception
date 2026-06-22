"""
Quantum Coherence as a Resource in Radical-Pair Dynamics
=========================================================
This script computes three measures of singlet-triplet (S-T) coherence as
a function of time for a radical pair evolving under three different master
equations, at a fixed field inclination theta.

The three formalisms compared are:

  * Haberkorn.
  * Jones & Hore.
  * Kominis.

For each formalism, one figure is produced showing four time traces:

  * Tr{rho}  -- radical-pair survival probability (population).

  * C_r      -- relative entropy of coherence:
                  C_r = S(rho_block) - S(rho),
                where rho_block = Q_S rho Q_S + Q_T rho Q_T is the
                S-T block-diagonal (incoherent) reference state and S is
                the von Neumann entropy. C_r is the minimal entropy cost
                of creating the coherence in rho from an incoherent state.

  * C_l1     -- l1-norm of coherence:
                  C_l1 = sum_{i!=j} |rho_ij|
                evaluated in the S-T reference basis {|S,nuc>, |T+,nuc>, ...}.
                This is the sum of all off-diagonal absolute values and is
                an easily computable lower bound on distillable coherence.

  * C_g      -- geometric coherence:
                  C_g = 1 - F^2(rho, rho_block),
                where F^2 is the squared Bures fidelity. C_g measures how
                far rho is from the nearest incoherent state in Bures geometry.

All three measures are evaluated on the normalised state rho/Tr{rho} (the
conditional state of a surviving pair) and then multiplied by Tr{rho} to
give an extensive quantity that tracks both the coherence per pair and the
number of surviving pairs simultaneously.

System
------
Two unpaired electrons (donor D, acceptor A) coupled to one nuclear spin
(I = 1/2), giving an 8-dimensional Hilbert space. Parameters follow
Guo et al. (2017), Sci. Rep. 7:5826: B0 = 46 uT, A_z = 6*gamma*B0,
kS = 1e4 s^-1, kT = 1e6 s^-1, theta = pi/4.

Usage
-----
    python coherence_measures.py

Output: coherence_measures_haberkorn.png / .csv
        coherence_measures_joneshore.png / .csv
        coherence_measures_kominis.png  / .csv

Requirements: numpy, scipy, matplotlib, pandas, tqdm
"""

import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


# =============================================================================
# 1. PHYSICAL PARAMETERS
# =============================================================================

GAMMA = 8.8e4    # Electron gyromagnetic ratio  [rad s^-1 uT^-1]
B0    = 46.0     # Geomagnetic field strength    [uT]
A_z   = 6 * GAMMA * B0    # Axial hyperfine coupling [rad s^-1]

kS = 1e4    # Singlet recombination rate  [s^-1]
kT = 1e6    # Triplet recombination rate  [s^-1]

THETA = np.pi / 4    # Fixed field inclination angle [rad]

# 15 us covers ~15 singlet lifetimes (1/kS = 100 us), enough for full decay.
# 2000 steps resolve the fast triplet transient (1/kT = 1 us) adequately.
T_MAX  = 15.0e-6    # [s]
N_STEPS = 2000
t_eval  = np.linspace(0, T_MAX, N_STEPS)


# =============================================================================
# 2. SPIN OPERATORS  (8x8 Hilbert space)
# =============================================================================

sx = np.array([[0,  1 ], [1,  0 ]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0 ]], dtype=complex)
sz = np.array([[1,  0 ], [0, -1 ]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def kron3(A, B, C):
    "Kronecker product of three 2x2 matrices into the 8x8 space."
    return np.kron(A, np.kron(B, C))


sDx = 0.5 * kron3(sx, I2, I2);  sDy = 0.5 * kron3(sy, I2, I2);  sDz = 0.5 * kron3(sz, I2, I2)
sAx = 0.5 * kron3(I2, sx, I2);  sAy = 0.5 * kron3(I2, sy, I2);  sAz = 0.5 * kron3(I2, sz, I2)
Iz  = 0.5 * kron3(I2, I2, sz)

# Singlet and triplet projectors
sD_dot_sA = sDx @ sAx + sDy @ sAy + sDz @ sAz
QS = 0.25 * np.eye(8, dtype=complex) - sD_dot_sA
QT = np.eye(8, dtype=complex) - QS

# Electron spin states (two-electron subspace)
_S  = np.array([0,  1/np.sqrt(2), -1/np.sqrt(2), 0], dtype=complex)
_Tp = np.array([1,  0,             0,             0], dtype=complex)
_T0 = np.array([0,  1/np.sqrt(2),  1/np.sqrt(2), 0], dtype=complex)
_Tm = np.array([0,  0,             0,             1], dtype=complex)

# Triplet sublevel projectors for the Kominis p_coh formula (8x8)
T_PROJECTORS = [
    np.kron(np.outer(_Tp, _Tp.conj()), I2),
    np.kron(np.outer(_T0, _T0.conj()), I2),
    np.kron(np.outer(_Tm, _Tm.conj()), I2),
]

# Reference basis for the l1-norm of coherence.
# The l1-norm is basis-dependent; we use the physical S-T basis extended
# with nuclear spin: {|S,up>, |S,dn>, |T+,up>, |T+,dn>, |T0,up>, |T0,dn>,
# |T-,up>, |T-,dn>}.  U_ST rotates from the computational basis to this
# ordered S-T basis, so rho in the S-T basis is U_ST_dag @ rho @ U_ST.
_nuc_up   = np.array([1, 0], dtype=complex)
_nuc_down = np.array([0, 1], dtype=complex)

U_ST = np.column_stack([
    np.kron(_S,  _nuc_up),   np.kron(_S,  _nuc_down),
    np.kron(_Tp, _nuc_up),   np.kron(_Tp, _nuc_down),
    np.kron(_T0, _nuc_up),   np.kron(_T0, _nuc_down),
    np.kron(_Tm, _nuc_up),   np.kron(_Tm, _nuc_down),
])
U_ST_dag = U_ST.conj().T


# =============================================================================
# 3. INITIAL STATE AND HAMILTONIAN
# =============================================================================

# Singlet electronic state, nuclear spin in completely mixed state
rho_0 = np.kron(np.outer(_S, _S.conj()), 0.5 * I2)

# Hamiltonian: isotropic Zeeman + axial hyperfine on acceptor electron
H = (  GAMMA * B0 * (np.sin(THETA) * (sDx + sAx) + np.cos(THETA) * (sDz + sAz))
      + A_z * (sAz @ Iz)  )


# =============================================================================
# 4. MASTER EQUATIONS
# =============================================================================

def _pcoh(rho):
    """
    Singlet-triplet coherence measure p_coh (Kritsotakis & Kominis 2014).
    Returns (p_coh, Tr{rho}).
    """
    tr = np.real(np.trace(rho))
    if tr < 1e-15:
        return 0.0, 0.0
    rho_ST = QS @ rho @ QT
    rho_TS = QT @ rho @ QS
    C = sum(np.sqrt(abs(np.trace(rho_ST @ P @ rho_TS))) for P in T_PROJECTORS)
    return float(np.clip((4.0 / 3.0) * C / tr, 0.0, 1.0)), tr


def L_haberkorn(t, y):
    """
    Haberkorn master equation:

      d rho/dt = -i[H, rho]
                 - (kS/2)(Q_S rho + rho Q_S)
                 - (kT/2)(Q_T rho + rho Q_T)
    """
    rho  = y.reshape((8, 8))
    drho = (  -1j * (H @ rho - rho @ H)
              - (kS / 2) * (QS @ rho + rho @ QS)
              - (kT / 2) * (QT @ rho + rho @ QT)  )
    return drho.flatten()


def L_jones_hore(t, y):
    """
    Jones-Hore master equation:

      d rho/dt = -i[H, rho]
                 - (kS/2)(Q_S rho + rho Q_S)
                 - (kT/2)(Q_T rho + rho Q_T)
                 - (kS+kT)/2 * (rho_ST + rho_TS)
    """
    rho    = y.reshape((8, 8))
    rho_ST = QS @ rho @ QT
    rho_TS = QT @ rho @ QS
    drho   = (  -1j * (H @ rho - rho @ H)
                - (kS / 2) * (QS @ rho + rho @ QS)
                - (kT / 2) * (QT @ rho + rho @ QT)
                - ((kS + kT) / 2) * (rho_ST + rho_TS)  )
    return drho.flatten()


def L_kominis(t, y):
    """
    Kominis master equation:

      d rho/dt = -i[H, rho]
                 - (kS+kT)/2 * (rho_ST + rho_TS)
                 - (1-p_coh)(kS rho_SS + kT rho_TT)
                 - (dr_S+dr_T)/Tr{rho} * (p_coh*rho_SS + p_coh*rho_TT
                                          + rho_ST + rho_TS)
    """
    rho          = y.reshape((8, 8))
    p_coh, tr    = _pcoh(rho)
    if tr < 1e-15:
        return np.zeros(64)
    rho_SS = QS @ rho @ QS
    rho_TT = QT @ rho @ QT
    rho_ST = QS @ rho @ QT
    rho_TS = QT @ rho @ QS
    rate_S = kS * np.real(np.trace(rho_SS))
    rate_T = kT * np.real(np.trace(rho_TT))
    drho   = (  -1j * (H @ rho - rho @ H)
                - ((kS + kT) / 2) * (rho_ST + rho_TS)
                - (1 - p_coh) * (kS * rho_SS + kT * rho_TT)
                - ((rate_S + rate_T) / tr)
                  * (p_coh * rho_SS + p_coh * rho_TT + rho_ST + rho_TS)  )
    return drho.flatten()


# =============================================================================
# 5. COHERENCE RESOURCE MEASURES
# =============================================================================

def _von_neumann_entropy(rho):
    "S(rho) = -Tr{rho log2 rho}, computed from eigenvalues."
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-15]
    return float(-np.sum(evals * np.log2(evals))) if len(evals) > 0 else 0.0


def _fidelity_squared(rho, sigma):
    """
    Squared Bures fidelity: F^2(rho, sigma) = [Tr sqrt(sqrt(rho) sigma sqrt(rho))]^2.

    A small regulariser (1e-15 * I) is added before each matrix square root to
    prevent numerical issues when either state has near-zero eigenvalues.
    """
    sqrt_rho = la.sqrtm(rho  + 1e-15 * np.eye(8))
    inner    = sqrt_rho @ (sigma + 1e-15 * np.eye(8)) @ sqrt_rho + 1e-15 * np.eye(8)
    return float(np.real(np.trace(la.sqrtm(inner)))**2)


def coherence_measures(rho):
    """
    Compute all three coherence measures for the (unnormalised) density matrix
    rho at a single time step.

    The measures are evaluated on the normalised conditional state rho_n =
    rho / Tr{rho}, then multiplied by Tr{rho} so the returned values track
    both the per-pair coherence and the surviving population simultaneously.

    Parameters
    ----------
    rho : ndarray, shape (8, 8) -- possibly unnormalised density matrix

    Returns
    -------
    pop : float -- Tr{rho}, radical-pair survival probability
    Cr  : float -- relative entropy of coherence * pop
    Cl1 : float -- l1-norm of coherence * pop
    Cg  : float -- geometric coherence * pop
    """
    pop = np.real(np.trace(rho))
    if pop < 1e-15:
        return pop, 0.0, 0.0, 0.0

    rho_n     = rho / pop
    rho_block = QS @ rho_n @ QS + QT @ rho_n @ QT    # incoherent reference state

    # Relative entropy of coherence: C_r = S(rho_block) - S(rho_n)
    cr = max(0.0, _von_neumann_entropy(rho_block) - _von_neumann_entropy(rho_n))

    # l1-norm of coherence in the S-T reference basis
    rho_off    = QS @ rho_n @ QT + QT @ rho_n @ QS
    rho_off_ST = U_ST_dag @ rho_off @ U_ST
    cl1        = float(np.sum(np.abs(rho_off_ST)))

    # Geometric coherence: C_g = 1 - F^2(rho_n, rho_block)
    cg = max(0.0, 1.0 - _fidelity_squared(rho_n, rho_block))

    return pop, cr * pop, cl1 * pop, cg * pop


# =============================================================================
# 6. RUN AND COLLECT
# =============================================================================

METHODS = {
    "Haberkorn":  L_haberkorn,
    "Jones-Hore": L_jones_hore,
    "Kominis":    L_kominis,
}

IVP_OPTS = dict(method='BDF', rtol=1e-8, atol=1e-10, max_step=1e-8)

all_results = {}

for name, rhs in METHODS.items():
    print(f"\nSolving {name} master equation.")
    sol = solve_ivp(rhs, [0, T_MAX], rho_0.flatten(), t_eval=t_eval, **IVP_OPTS)

    pop_arr, Cr_arr, Cl1_arr, Cg_arr = [], [], [], []

    for i in tqdm(range(len(sol.t)), desc=f"  {name} coherence", ncols=80):
        rho_t = sol.y[:, i].reshape(8, 8)
        pop, cr, cl1, cg = coherence_measures(rho_t)
        pop_arr.append(pop);  Cr_arr.append(cr)
        Cl1_arr.append(cl1);  Cg_arr.append(cg)

    all_results[name] = {
        "pop": np.array(pop_arr),
        "Cr":  np.array(Cr_arr),
        "Cl1": np.array(Cl1_arr),
        "Cg":  np.array(Cg_arr),
    }


# =============================================================================
# 7. PLOT AND SAVE
# =============================================================================

t_us = t_eval * 1e6    # convert to microseconds for the x-axis

FILE_STEMS = {
    "Haberkorn":  "coherence_measures_haberkorn",
    "Jones-Hore": "coherence_measures_joneshore",
    "Kominis":    "coherence_measures_kominis",
}

for name, res in all_results.items():
    stem = FILE_STEMS[name]

    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.plot(t_us, res["pop"], color='black',     lw=2.5, label=r'Population $\mathrm{Tr}(\hat{\rho})$')
    ax.plot(t_us, res["Cr"],  color='tab:blue',  lw=2.5, label=r'Relative entropy $C_r$')
    ax.plot(t_us, res["Cl1"], color='tab:purple', lw=2.5, label=r'$\ell_1$-norm $C_{\ell_1}$')
    ax.plot(t_us, res["Cg"],  color='tab:green', lw=3.0, label=r'Geometric coherence $C_g$')

    ax.set_title(name, fontsize=14)
    ax.set_xlabel(r'Time, $t$ ($\mu$s)', fontsize=14)
    ax.set_ylabel('Coherence resource', fontsize=14)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.5)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)

    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=300)
    plt.close(fig)

    pd.DataFrame({
        'time_us':    t_us,
        'Population': res["pop"],
        'C_r':        res["Cr"],
        'C_l1':       res["Cl1"],
        'C_g':        res["Cg"],
    }).to_csv(f"{stem}.csv", index=False)
