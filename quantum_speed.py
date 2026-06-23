"""
Instantaneous Quantum Speed of the Radical-Pair Compass
========================================================
This script computes and plots the instantaneous quantum speed v(t) of a
radical pair evolving under three master-equation formalisms, at a fixed
geomagnetic field inclination theta.

The quantum speed is defined as:

    v(t) = (1/2) * sqrt( QFI_time(t) )

where QFI_time(t) is the quantum Fisher information computed with respect to
the time parameter, i.e. using the Liouvillian L[rho(t)] as the generator:

    QFI_time(t) = 2 * sum_{i,j: p_i+p_j > 0}  |<i| L[rho] |j>|^2 / (p_i + p_j)

This is the Bures metric speed: it measures how fast the state rho(t) is
moving through the space of density matrices at time t, in the sense of the
distinguishability distance.  For a unitary evolution v(t) = Delta H (energy
uncertainty), recovering the Mandelstam-Tamm bound.  For open systems,
dissipation generally reduces v(t) relative to the unitary case.

The three formalisms compared are:

  * Haberkorn.
  * Jones & Hore.
  * Kominis.

System
------
Two unpaired electrons (donor D, acceptor A) coupled to one nuclear spin
(I = 1/2), 8-dimensional Hilbert space.  Parameters follow Guo et al. (2017):
B0 = 46 uT, A_z = 6*gamma*B0, kS = 1e4 s^-1, kT = 1e6 s^-1, theta = pi/4.

Usage
-----
    python velocity.py

Output: quantum_speed.png

Requirements: numpy, scipy, matplotlib
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# =============================================================================
# 1. PHYSICAL PARAMETERS
# =============================================================================

GAMMA = 8.8e4    # Electron gyromagnetic ratio  [rad s^-1 uT^-1]
B0    = 46.0     # Geomagnetic field strength    [uT]
A_z   = 6 * GAMMA * B0    # Axial hyperfine coupling [rad s^-1]

kS = 1e4    # Singlet recombination rate  [s^-1]
kT = 1e6    # Triplet recombination rate  [s^-1]

THETA = np.pi / 4    # Fixed field inclination angle [rad]

# 15 us covers the full singlet lifetime; 500 steps are enough to resolve
# v(t) smoothly since it varies on the triplet timescale (~1 us).
T_MAX   = 15e-6    # [s]
N_STEPS = 500
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

sD_dot_sA = sDx @ sAx + sDy @ sAy + sDz @ sAz
QS = 0.25 * np.eye(8, dtype=complex) - sD_dot_sA
QT = np.eye(8, dtype=complex) - QS

_Tp = np.array([1, 0, 0, 0])
_T0 = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0])
_Tm = np.array([0, 0, 0, 1])
T_PROJECTORS = [
    np.kron(np.outer(_Tp, _Tp), I2),
    np.kron(np.outer(_T0, _T0), I2),
    np.kron(np.outer(_Tm, _Tm), I2),
]


# =============================================================================
# 3. INITIAL STATE AND HAMILTONIAN
# =============================================================================

_S    = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0], dtype=complex)
rho_0 = np.kron(np.outer(_S, _S.conj()), 0.5 * I2)

H = (  GAMMA * B0 * (np.sin(THETA) * (sDx + sAx) + np.cos(THETA) * (sDz + sAz))
      + A_z * (sAz @ Iz)  )


# =============================================================================
# 4. LIOUVILLIANS
# =============================================================================

def _pcoh(rho):
    "Singlet-triplet coherence measure p_coh (Kritsotakis & Kominis 2014)."
    tr = np.real(np.trace(rho))
    if tr < 1e-15:
        return 0.0, 0.0
    rho_ST = QS @ rho @ QT
    rho_TS = QT @ rho @ QS
    C = sum(np.sqrt(abs(np.trace(rho_ST @ P @ rho_TS))) for P in T_PROJECTORS)
    return float(np.clip((4.0 / 3.0) * C / tr, 0.0, 1.0)), tr


def L_haberkorn(rho):
    return (  -1j * (H @ rho - rho @ H)
              - (kS / 2) * (QS @ rho + rho @ QS)
              - (kT / 2) * (QT @ rho + rho @ QT)  )


def L_jones_hore(rho):
    rho_ST = QS @ rho @ QT
    rho_TS = QT @ rho @ QS
    return (  L_haberkorn(rho)
              - ((kS + kT) / 2) * (rho_ST + rho_TS)  )


def L_kominis(rho):
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


LIOUVILLIANS = {
    "Haberkorn":  L_haberkorn,
    "Jones-Hore": L_jones_hore,
    "Kominis":    L_kominis,
}


# =============================================================================
# 5. QUANTUM SPEED
# =============================================================================

def _qfi_time(rho, L):
    """
    Quantum Fisher information with respect to time at a single state rho.

    Uses the Liouvillian L[rho] as the generator (i.e. d rho/dt), evaluated
    via the standard spectral formula:

        QFI_time = 2 * sum_{i,j: p_i+p_j > 0}  |<i| L[rho] |j>|^2 / (p_i + p_j)

    The generator is symmetrised before use to correct small Hermiticity
    violations introduced by the open-system terms.

    Returns 0 if the population is effectively zero.
    """
    tr = np.real(np.trace(rho))
    if tr < 1e-12:
        return 0.0

    drho = L(rho)
    drho = 0.5 * (drho + drho.conj().T)    # enforce Hermiticity

    vals, vecs = np.linalg.eigh(rho)
    vals = np.clip(vals, 0, None)

    qfi = 0.0
    for i in range(8):
        for j in range(8):
            if vals[i] + vals[j] > 1e-10:
                mel  = vecs[:, i].conj() @ drho @ vecs[:, j]
                qfi += 2 * abs(mel)**2 / (vals[i] + vals[j])
    return float(np.real(qfi))


def compute_speed(L):
    """
    Integrate the master equation and return v(t) = (1/2) sqrt(QFI_time(t)).

    The result is expressed in us^-1 (by dividing by 1e6) so the plot
    x-axis (in us) and y-axis share a compatible scale.

    Parameters
    ----------
    L : callable -- Liouvillian rho -> drho (matrix, not flattened)

    Returns
    -------
    v_t : ndarray, shape (N_STEPS,)  [us^-1]
    """
    sol = solve_ivp(
        lambda t, y: L(y.reshape((8, 8))).flatten(),
        [0, T_MAX], rho_0.flatten(),
        t_eval=t_eval, method='BDF', rtol=1e-10, atol=1e-12,
    )
    rho_t = sol.y.T.reshape((N_STEPS, 8, 8))

    v_t = np.array([
        0.5 * np.sqrt(_qfi_time(0.5 * (rho + rho.conj().T), L))
        for rho in rho_t
    ])
    return v_t / 1e6    # convert s^-1 -> us^-1


# =============================================================================
# 6. MAIN
# =============================================================================

if __name__ == "__main__":

    STYLES = {
        "Haberkorn":  ("blue",  "--"),
        "Jones-Hore": ("green", "-."),
        "Kominis":    ("red",   "-"),
    }

    t_us = t_eval * 1e6

    fig, ax = plt.subplots(figsize=(10, 6.5))

    for name, L in LIOUVILLIANS.items():
        print(f"  {name}...", end=" ", flush=True)
        v_t = compute_speed(L)
        col, ls = STYLES[name]
        ax.plot(t_us, v_t, color=col, ls=ls, lw=2.5, label=name)

    ax.set_xlabel(r'Time, $t$ ($\mu$s)', fontsize=14)
    ax.set_ylabel(r'Instantaneous speed, $v(t)$ ($\mu$s$^{-1}$)', fontsize=14)
    ax.set_xlim(0, t_us[-1])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)

    fig.tight_layout()
    fig.savefig('quantum_speed.png', dpi=300)
    plt.close(fig)
