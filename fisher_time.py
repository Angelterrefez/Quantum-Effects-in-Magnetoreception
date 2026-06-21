"""
Time-Domain Fisher Information of the Radical-Pair Compass
===========================================================
This script computes the quantum Fisher information (QFI) and classical Fisher
information (CFI) of the instantaneous radical-pair (RP) density matrix rho(t)
as a function of time, at a fixed geomagnetic field inclination angle theta.

Rather than working with the time-averaged steady state (as in QFI.py), here
we track the Fisher information along the full time evolution of a single RP
from creation to recombination. This reveals how the directional information
encoded in the spin state builds up and decays as the pair evolves.

The same three master-equation formalisms are compared:

  * Haberkorn.
  * Jones & Hore.
  * Kominis.

At each time step the QFI is computed from the spectral decomposition of
rho(t), and the CFI is evaluated via the error-propagation formula for
three observables:

  * Q_S   -- singlet projector (singlet yield proxy)
  * S^2   -- total spin squared (S_D + S_A)^2
  * S_z^2 -- square of total z-spin, related to fluctuation of magnetic moment

The derivative drho/dtheta needed for both QFI and CFI is obtained by a
centred finite difference: three parallel integrations are run at theta,
theta + dtheta, and theta - dtheta.

System
------
Two unpaired electrons (donor D, acceptor A) coupled to one nuclear spin
(I = 1/2), giving an 8-dimensional Hilbert space.

Basis ordering: |uu u>, |uu d>, |ud u>, |ud d>, |du u>, |du d>, |dd u>, |dd d>
(first two indices = electron spins D and A; last = nuclear spin I).

Usage
-----
    python fisher_time.py

Output: four PNG plots of QFI and CFI vs time [us].

Requirements: numpy, scipy, matplotlib
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# =============================================================================
# 1. PHYSICAL PARAMETERS
# =============================================================================

GAMMA = 8.8e4   # Electron gyromagnetic ratio  [rad s^-1 uT^-1]
B0    = 46.0    # Earth's magnetic field strength [uT]

# Axial hyperfine coupling (A_x = A_y = 0).
# The asymmetry between the two electrons -- one free, one coupled -- is what
# drives the S-T mixing that makes the compass directionally sensitive.
A_z = 6 * GAMMA * B0    # [rad s^-1]

# Recombination rates. kT >> kS means triplet pairs decay fast and singlet
# pairs are long-lived, a relevant asymmetric regime.
kS = 1e4    # Singlet recombination rate  [s^-1]
kT = 1e6    # Triplet recombination rate  [s^-1]

# Set fixed inclination angle.
THETA = np.pi / 4

# Time grid. 15 us covers ~15 singlet lifetimes (1/kS = 100 us), capturing
# >99% of the dynamics. 500 points give enough resolution to see the fast
# initial transient driven by the triplet channel (1/kT = 1 us).
T_MAX       = 15e-6    # [s]
N_STEPS     = 500
t_eval      = np.linspace(0, T_MAX, N_STEPS)


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


# Electron spin operators for donor (D) and acceptor (A), and nuclear spin (I)
sDx = 0.5 * kron3(sx, I2, I2);  sDy = 0.5 * kron3(sy, I2, I2);  sDz = 0.5 * kron3(sz, I2, I2)
sAx = 0.5 * kron3(I2, sx, I2);  sAy = 0.5 * kron3(I2, sy, I2);  sAz = 0.5 * kron3(I2, sz, I2)
Ix  = 0.5 * kron3(I2, I2, sx);  Iy  = 0.5 * kron3(I2, I2, sy);  Iz  = 0.5 * kron3(I2, I2, sz)

# Singlet and triplet projectors: Q_S = 1/4 - s_D.s_A,  Q_T = 1 - Q_S
s_D_dot_s_A = sDx @ sAx + sDy @ sAy + sDz @ sAz
QS = 0.25 * np.eye(8, dtype=complex) - s_D_dot_s_A
QT = np.eye(8, dtype=complex) - QS

# Observables for CFI calculations
S_tot_x = sDx + sAx
S_tot_y = sDy + sAy
S_tot_z = sDz + sAz
S2_op  = S_tot_x @ S_tot_x + S_tot_y @ S_tot_y + S_tot_z @ S_tot_z
Sz2_op = S_tot_z @ S_tot_z

# Triplet sublevel projectors needed by the Kominis p_coh formula
_T_p = np.array([1, 0, 0, 0])
_T_0 = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0])
_T_m = np.array([0, 0, 0, 1])
T_PROJECTORS = [
    np.kron(np.outer(_T_p, _T_p), I2),
    np.kron(np.outer(_T_0, _T_0), I2),
    np.kron(np.outer(_T_m, _T_m), I2),
]


# =============================================================================
# 3. INITIAL STATE
# =============================================================================
# Each RP is created in the singlet electron state; the nuclear spin starts
# in a completely mixed state (equal probability of up and down).

state_S = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0], dtype=complex)
rho_S   = np.outer(state_S, state_S.conj())
rho_0   = np.kron(rho_S, 0.5 * I2)    # rho_e(0) x (I/2)


# =============================================================================
# 4. HAMILTONIAN
# =============================================================================

def build_hamiltonian(B0_mag, theta):
    """
    Spin Hamiltonian for a given field strength and inclination:

        H = gamma * B0 * [sin(theta)(S_Dx + S_Ax) + cos(theta)(S_Dz + S_Az)]
          + A_z * S_Az * I_z

    The Zeeman term couples both electrons equally to the external field.
    The hyperfine term couples only the acceptor electron to the nuclear spin,
    breaking the symmetry and providing the directional sensitivity.

    Parameters
    ----------
    B0_mag : float -- field strength [uT]
    theta  : float -- inclination angle [rad]
    """
    H_zeeman    = GAMMA * B0_mag * (np.sin(theta) * (sDx + sAx) + np.cos(theta) * (sDz + sAz))
    H_hyperfine = A_z * (sAz @ Iz)
    return H_zeeman + H_hyperfine


# =============================================================================
# 5. MASTER EQUATIONS (Liouvillians)
# =============================================================================
# Note: unlike QFI.py, the Liouvillians here return the 8x8 matrix directly
# (not flattened), because the time-domain loop works with the matrix form
# more naturally when computing observables step by step.

def _pcoh(rho):
    """
    Singlet-triplet coherence measure p_coh from Kritsotakis & Kominis (2014).

    p_coh in [0, 1]:
      0 -- maximally incoherent mixture of S and T
      1 -- maximally coherent superposition

    Returns (p_coh, Tr{rho}).
    """
    tr = np.real(np.trace(rho))
    if tr < 1e-15:
        return 0.0, 0.0
    rho_ST = QS @ rho @ QT
    rho_TS = QT @ rho @ QS
    C = sum(np.sqrt(abs(np.trace(rho_ST @ P @ rho_TS))) for P in T_PROJECTORS)
    return float(np.clip((4.0 / 3.0) * C / tr, 0.0, 1.0)), tr


def L_haberkorn(rho, H):
    """
    Haberkorn master equation:

      d rho/dt = -i[H, rho]
                 - (kS/2)(Q_S rho + rho Q_S)
                 - (kT/2)(Q_T rho + rho Q_T)
    """
    return (  -1j * (H @ rho - rho @ H)
              - (kS / 2) * (QS @ rho + rho @ QS)
              - (kT / 2) * (QT @ rho + rho @ QT)  )


def L_jones_hore(rho, H):
    """
    Jones-Hore master equation:

      d rho/dt = -i[H, rho]
                 - (kS/2)(Q_S rho + rho Q_S)
                 - (kT/2)(Q_T rho + rho Q_T)
                 - (kS+kT)/2 * (rho_ST + rho_TS)
    """
    rho_ST = QS @ rho @ QT
    rho_TS = QT @ rho @ QS
    return (  -1j * (H @ rho - rho @ H)
              - (kS / 2) * (QS @ rho + rho @ QS)
              - (kT / 2) * (QT @ rho + rho @ QT)
              - ((kS + kT) / 2) * (rho_ST + rho_TS)  )


def L_kominis(rho, H):
    """
    Kominis master equation:

      d rho/dt = -i[H, rho]
                 - (kS+kT)/2 * (rho_ST + rho_TS)
                 - (1-p_coh)(kS rho_SS + kT rho_TT)
                 - (dr_S+dr_T)/Tr{rho} * (p_coh*rho_SS + p_coh*rho_TT
                                          + rho_ST + rho_TS)

    The early-exit guard prevents the ODE solver from amplifying floating-point
    noise once the pair population has effectively vanished.
    """
    p_coh, tr = _pcoh(rho)
    if tr < 1e-15:
        return np.zeros_like(rho)

    rho_SS = QS @ rho @ QS
    rho_TT = QT @ rho @ QT
    rho_ST = QS @ rho @ QT
    rho_TS = QT @ rho @ QS

    rate_S = kS * np.real(np.trace(rho_SS))
    rate_T = kT * np.real(np.trace(rho_TT))

    return (  -1j * (H @ rho - rho @ H)
              - ((kS + kT) / 2) * (rho_ST + rho_TS)
              - (1 - p_coh) * (kS * rho_SS + kT * rho_TT)
              - ((rate_S + rate_T) / tr)
                * (p_coh * rho_SS + p_coh * rho_TT + rho_ST + rho_TS)  )


# =============================================================================
# 6. TIME EVOLUTION
# =============================================================================

def evolve(H, method):
    """
    Integrate the master equation from t=0 to T_MAX and return rho(t) on
    the t_eval grid.

    Parameters
    ----------
    H      : ndarray, shape (8, 8) -- Hamiltonian
    method : str -- "Haberkorn", "Jones-Hore", or "Kominis"

    Returns
    -------
    sol_y : ndarray, shape (64, N_STEPS) -- flattened rho at each time point
    """
    LIOUVILLIANS = {
        "Haberkorn":  L_haberkorn,
        "Jones-Hore": L_jones_hore,
        "Kominis":    L_kominis,
    }
    L = LIOUVILLIANS[method]

    def rhs(t, y):
        return L(y.reshape((8, 8)), H).flatten()

    sol = solve_ivp(
        rhs, [0, T_MAX], rho_0.flatten(),
        method='BDF', t_eval=t_eval, rtol=1e-10, atol=1e-12,
    )
    return sol.y    # shape (64, N_STEPS)


# =============================================================================
# 7. TIME-DOMAIN FISHER INFORMATION
# =============================================================================

def compute_fisher_time_series(method, dtheta=1e-4):
    """
    Compute QFI(t) and CFI(t) for each time step in t_eval.

    Three integrations are run (at theta, theta+dtheta, theta-dtheta) and the
    derivative drho/dtheta is approximated by a centred finite difference at
    each time step independently.

    At each step the QFI uses the standard spectral formula:

        QFI(t) = 2 * sum_{i,j: p_i+p_j > 0}  |<i| drho/dtheta |j>|^2 / (p_i + p_j)

    and the CFI for observable O uses:

        CFI(t, O) = (d<O>/dtheta)^2 / Var_rho(O)

    Both are set to zero once the pair population Tr{rho} drops below a
    numerical threshold, since the Fisher information is undefined for a
    zero-population state.

    Parameters
    ----------
    method : str   -- "Haberkorn", "Jones-Hore", or "Kominis"
    dtheta : float -- finite-difference step [rad]

    Returns
    -------
    qfi_t    : list[float] -- QFI at each time step
    cfi_QS_t : list[float] -- CFI for Q_S
    cfi_S2_t : list[float] -- CFI for S^2
    cfi_Sz2_t: list[float] -- CFI for S_z^2
    """
    H   = build_hamiltonian(B0, THETA)
    H_p = build_hamiltonian(B0, THETA + dtheta)
    H_m = build_hamiltonian(B0, THETA - dtheta)

    print(f"  Integrating {method}", end=" ", flush=True)
    sol   = evolve(H,   method)
    sol_p = evolve(H_p, method)
    sol_m = evolve(H_m, method)

    qfi_t, cfi_QS_t, cfi_S2_t, cfi_Sz2_t = [], [], [], []

    for k in range(N_STEPS):
        rho   = 0.5 * (sol[:,   k].reshape(8, 8) + sol[:,   k].reshape(8, 8).conj().T)
        rho_p = 0.5 * (sol_p[:, k].reshape(8, 8) + sol_p[:, k].reshape(8, 8).conj().T)
        rho_m = 0.5 * (sol_m[:, k].reshape(8, 8) + sol_m[:, k].reshape(8, 8).conj().T)

        tr   = np.real(np.trace(rho))
        tr_p = np.real(np.trace(rho_p))
        tr_m = np.real(np.trace(rho_m))

        # Fisher information is undefined once the population has vanished
        if tr < 1e-12 or tr_p < 1e-12 or tr_m < 1e-12:
            qfi_t.append(0.0)
            cfi_QS_t.append(0.0)
            cfi_S2_t.append(0.0)
            cfi_Sz2_t.append(0.0)
            continue

        drho = (rho_p - rho_m) / (2 * dtheta)

        # QFI via spectral decomposition
        vals, vecs = np.linalg.eigh(rho)
        vals = np.clip(vals, 0, None)

        qfi = 0.0
        for i in range(8):
            for j in range(8):
                if vals[i] + vals[j] > 1e-10:
                    mel  = vecs[:, i].conj() @ drho @ vecs[:, j]
                    qfi += 2 * abs(mel)**2 / (vals[i] + vals[j])
        qfi_t.append(float(np.real(qfi)))

        # CFI via error propagation for a given observable
        def cfi(O):
            exp_O  = np.real(np.trace(rho @ O))
            exp_O2 = np.real(np.trace(rho @ O @ O))
            var    = exp_O2 - exp_O**2
            d_exp  = np.real(np.trace(drho @ O))
            if var > 1e-11:
                return d_exp**2 / var
            return 0.0

        cfi_QS_t.append(cfi(QS))
        cfi_S2_t.append(cfi(S2_op))
        cfi_Sz2_t.append(cfi(Sz2_op))

    return qfi_t, cfi_QS_t, cfi_S2_t, cfi_Sz2_t


# =============================================================================
# 8. PLOTTING
# =============================================================================

def make_plot(t_us, y_H, y_JH, y_K, ylabel, filename):
    """
    Plot Fisher information as a function of time for the three formalisms.

    Parameters
    ----------
    t_us             : ndarray -- time axis [us]
    y_H, y_JH, y_K  : array-like -- Haberkorn, Jones-Hore, Kominis traces
    ylabel           : str -- y-axis label
    filename         : str -- output file path
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.plot(t_us, y_H,  color='blue',  lw=2.5, ls='--', label='Haberkorn')
    ax.plot(t_us, y_JH, color='green', lw=2.5, ls='-.', label='Jones-Hore')
    ax.plot(t_us, y_K,  color='red',   lw=2.5,           label='Kominis')

    ax.set_xlabel(r'Time, $t$ ($\mu$s)', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_xlim(0, t_us[-1])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    ax.legend(fontsize=12)

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


# =============================================================================
# 9. MAIN
# =============================================================================

if __name__ == "__main__":

    results = {
        m: {"qfi": [], "cfi_QS": [], "cfi_S2": [], "cfi_Sz2": []}
        for m in ("Haberkorn", "Jones-Hore", "Kominis")
    }

    for method in ("Haberkorn", "Jones-Hore", "Kominis"):
        qfi, cQS, cS2, cSz2 = compute_fisher_time_series(method)
        results[method]["qfi"]    = qfi
        results[method]["cfi_QS"] = cQS
        results[method]["cfi_S2"] = cS2
        results[method]["cfi_Sz2"]= cSz2

    t_us = t_eval * 1e6    # convert to microseconds for the x-axis

    make_plot(
        t_us,
        results["Haberkorn"]["qfi"],
        results["Jones-Hore"]["qfi"],
        results["Kominis"]["qfi"],
        ylabel=r'QFI  $1/\Delta^2\theta$',
        filename="QFI_time_pi4.png",
    )

    make_plot(
        t_us,
        results["Haberkorn"]["cfi_QS"],
        results["Jones-Hore"]["cfi_QS"],
        results["Kominis"]["cfi_QS"],
        ylabel=r'CFI ($Q_S$ measurement)  $1/\Delta^2\theta$',
        filename="CFI_QS_time_pi4.png",
    )

    make_plot(
        t_us,
        results["Haberkorn"]["cfi_S2"],
        results["Jones-Hore"]["cfi_S2"],
        results["Kominis"]["cfi_S2"],
        ylabel=r'CFI ($\hat{S}^2$ measurement)  $1/\Delta^2\theta$',
        filename="CFI_S2_time_pi4.png",
    )

    make_plot(
        t_us,
        results["Haberkorn"]["cfi_Sz2"],
        results["Jones-Hore"]["cfi_Sz2"],
        results["Kominis"]["cfi_Sz2"],
        ylabel=r'CFI ($\hat{S}_z^2$ measurement)  $1/\Delta^2\theta$',
        filename="CFI_Sz2_time_pi4.png",
    )