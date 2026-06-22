"""
Quantum Speed Limit of the Radical-Pair Compass
================================================
This script computes the quantum speed limit (QSL) time for a radical pair
evolving under four dynamical models, at a fixed geomagnetic field inclination
theta, and tracks it as a function of the elapsed evolution time t.

The QSL time tau_QSL(t) is the tightest lower bound on the time needed to
drive the system from its initial state rho_0 to the state rho(t) it has
actually reached.  The bound used here follows from the Bures angle distance
and the operator-norm speed of the generator (Deffner & Lutz 2013):

    tau_QSL(t) = sin^2[ L(rho_0, rho(t)) ] / Lambda(t)

where:

  L(rho_0, rho(t)) = arccos( F(rho_0, rho(t)) )   -- Bures angle
  F(rho_0, rho(t)) = Tr [sqrt( sqrt(rho_0) rho(t) sqrt(rho_0) )]   -- fidelity amplitude
  Lambda(t) = (1/t) integral_0^t ||L[rho(s)]|| ds  -- time-averaged operator norm of the Liouvillian

When tau_QSL(t) = t the evolution is at the speed limit (maximally efficient).
When tau_QSL(t) < t there is quantum speed-up potential: the system could in
principle have reached rho(t) faster under a different generator.

Four models are compared:

  * Unitary        -- closed quantum system, no recombination.
  * Haberkorn      
  * Jones & Hore  
  * Kominis        

Three figures are produced:

  Figure 1 -- tau_QSL(t) vs t, with the diagonal y=x as a reference.
              Points below the diagonal indicate speed-up potential.
  Figure 2 -- Time-averaged Liouvillian norm Lambda(t) vs t.
  Figure 3 -- Bures fidelity F(rho_0, rho(t)) vs t.

System
------
Two unpaired electrons (donor D, acceptor A) coupled to one nuclear spin
(I = 1/2), 8-dimensional Hilbert space.  Parameters follow Guo et al. (2017):
B0 = 46 uT, A_z = 6*gamma*B0, kS = 1e4 s^-1, kT = 1e6 s^-1.

Usage
-----
    python qsl_time.py

Output: qsl_tau.png, qsl_lambda.png, qsl_fidelity.png

Requirements: numpy, scipy, matplotlib, pandas, tqdm
"""

import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:
    from scipy.integrate import cumtrapz as cumulative_trapezoid    # scipy < 1.8

import matplotlib.pyplot as plt
from tqdm import tqdm


# =============================================================================
# 1. PHYSICAL PARAMETERS
# =============================================================================

GAMMA = 8.8e4    # Electron gyromagnetic ratio  [rad s^-1 uT^-1]
B0    = 46.0     # Geomagnetic field strength    [uT]
A_z   = 6 * GAMMA * B0    # Axial hyperfine coupling [rad s^-1]

kS = 1e4    # Singlet recombination rate  [s^-1]
kT = 1e6    # Triplet recombination rate  [s^-1]

THETA = np.pi / 4

# 15 us covers the full singlet lifetime; 5000 steps resolve the fast
# triplet transient (1/kT = 1 us) and keep the trapezoidal integration
# of the Liouvillian norm accurate.
T_MAX  = 15.0e-6    # [s]
N_STEPS = 5000
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

# Triplet sublevel projectors for the Kominis p_coh formula
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

# Pre-compute sqrt(rho_0) once; it is reused at every time step to evaluate
# the Bures fidelity F(rho_0, rho(t)) = Tr sqrt( sqrt(rho_0) rho(t) sqrt(rho_0) ).
sqrt_rho_0 = la.sqrtm(rho_0)

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


def L_unitary(rho):
    "Closed-system (unitary) evolution: d rho/dt = -i[H, rho]."
    return -1j * (H @ rho - rho @ H)


def L_haberkorn(rho):
    """
    Haberkorn master equation:

      d rho/dt = -i[H, rho]
                 - (kS/2)(Q_S rho + rho Q_S)
                 - (kT/2)(Q_T rho + rho Q_T)
    """
    return (  L_unitary(rho)
              - (kS / 2) * (QS @ rho + rho @ QS)
              - (kT / 2) * (QT @ rho + rho @ QT)  )


def L_jones_hore(rho):
    """
    Jones-Hore master equation:

      d rho/dt = -i[H, rho]
                 - (kS/2)(Q_S rho + rho Q_S)
                 - (kT/2)(Q_T rho + rho Q_T)
                 - (kS+kT)/2 * (rho_ST + rho_TS)
    """
    rho_ST = QS @ rho @ QT
    rho_TS = QT @ rho @ QS
    return (  L_haberkorn(rho)
              - ((kS + kT) / 2) * (rho_ST + rho_TS)  )


def L_kominis(rho):
    """
    Kominis master equation:

      d rho/dt = -i[H, rho]
                 - (kS+kT)/2 * (rho_ST + rho_TS)
                 - (1-p_coh)(kS rho_SS + kT rho_TT)
                 - (dr_S+dr_T)/Tr{rho} * (p_coh*rho_SS + p_coh*rho_TT
                                          + rho_ST + rho_TS)
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
    return (  L_unitary(rho)
              - ((kS + kT) / 2) * (rho_ST + rho_TS)
              - (1 - p_coh) * (kS * rho_SS + kT * rho_TT)
              - ((rate_S + rate_T) / tr)
                * (p_coh * rho_SS + p_coh * rho_TT + rho_ST + rho_TS)  )


LIOUVILLIANS = {
    "Unitary":    L_unitary,
    "Haberkorn":  L_haberkorn,
    "Jones-Hore": L_jones_hore,
    "Kominis":    L_kominis,
}


# =============================================================================
# 5. QSL DIAGNOSTICS
# =============================================================================

def _operator_norm(A):
    "Largest singular value of A, i.e. the operator (spectral) norm ||A||."
    return float(np.max(la.svd(A, compute_uv=False)))


def _bures_fidelity(rho_t):
    """
    Bures fidelity amplitude F(rho_0, rho_t) = Tr sqrt( sqrt(rho_0) rho_t sqrt(rho_0) ).

    sqrt(rho_0) is pre-computed at module level.  The result is clipped to
    [0, 1] to guard against tiny negative values from numerical noise.
    """
    core = sqrt_rho_0 @ rho_t @ sqrt_rho_0
    return float(np.clip(np.real(np.trace(la.sqrtm(core))), 0.0, 1.0))


def compute_qsl_series(L):
    """
    Integrate the master equation and compute the QSL diagnostics.

    At each time step three quantities are returned:

      tau_QSL(t) = sin^2(arccos F) / Lambda(t)
                 = (1 - F^2) / Lambda(t)    [in seconds]

        This is the QSL time: the minimum time the system would have needed
        to reach rho(t) from rho_0 under the given generator.

      Lambda(t)  = (1/t) * integral_0^t ||L[rho(s)]|| ds
                 = time-averaged operator norm of the generator [s^-1]

      F(t)       = Bures fidelity amplitude F(rho_0, rho(t))

    Parameters
    ----------
    L : callable -- Liouvillian rho -> drho (matrix form, not flattened)

    Returns
    -------
    tau_QSL : ndarray, shape (N_STEPS,)  [s]
    Lambda_t : ndarray, shape (N_STEPS,)  [s^-1]
    fidelity : ndarray, shape (N_STEPS,)  [dimensionless]
    """
    sol = solve_ivp(
        lambda t, y: L(y.reshape((8, 8))).flatten(),
        [0, T_MAX], rho_0.flatten(),
        t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-10, max_step=1e-8,
    )
    rho_t = sol.y.T.reshape((N_STEPS, 8, 8))

    # Instantaneous operator norm and fidelity at each time step
    norm_L   = np.array([_operator_norm(L(rho_t[i]))    for i in range(N_STEPS)])
    fidelity = np.array([_bures_fidelity(rho_t[i])      for i in range(N_STEPS)])

    # Cumulative time integral of ||L||, then divide by t to get Lambda(t).
    # cumulative_trapezoid returns an array of length N_STEPS with initial=0.
    cum_norm = cumulative_trapezoid(norm_L, t_eval, initial=0.0)

    # Lambda(t) = (1/t) * integral; t[0]=0 is handled by the where clause.
    Lambda_t = np.where(t_eval > 0, cum_norm / t_eval, 0.0)

    # QSL time: sin^2(Bures angle) / Lambda(t).
    # sin^2(arccos F) = 1 - F^2.
    sin2_bures = 1.0 - fidelity**2
    tau_QSL    = np.where(Lambda_t > 1e-18, sin2_bures / Lambda_t, 0.0)

    return tau_QSL, Lambda_t, fidelity


# =============================================================================
# 6. RUN
# =============================================================================

if __name__ == "__main__":

    results = {}

    print(f"Computing QSL time series at theta over {T_MAX*1e6:.0f} us.")
    for name, L in LIOUVILLIANS.items():
        print(f"  {name}...", end=" ", flush=True)
        tau, lam, fid = compute_qsl_series(L)
        results[name] = {"tau_QSL": tau, "Lambda": lam, "Fidelity": fid}

    # Convert to microseconds for axis labels
    t_us     = t_eval * 1e6
    tau_us   = {n: r["tau_QSL"] * 1e6 for n, r in results.items()}
    lam_us   = {n: r["Lambda"]  / 1e6 for n, r in results.items()}    # s^-1 -> us^-1
    fid      = {n: r["Fidelity"]      for n, r in results.items()}

    STYLES = {
        "Unitary":    ("black",  ":"),
        "Haberkorn":  ("blue",   "--"),
        "Jones-Hore": ("green",  "-."),
        "Kominis":    ("red",    "-"),
    }

    # -------------------------------------------------------------------------
    # Figure 1: QSL time
    # The diagonal y = x is the physical bound: tau_QSL <= t always.
    # A curve below the diagonal means the generator is not operating at its
    # speed limit and the same state could in principle be reached faster.
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 6.5))
    ax1.plot(t_us, t_us, color='purple', lw=2.5, label='Actual elapsed time $t$')
    for name, (col, ls) in STYLES.items():
        ax1.plot(t_us, tau_us[name], color=col, ls=ls, lw=2.5, label=name)
    ax1.set_xlabel(r'Evolution time, $t$ ($\mu$s)', fontsize=14)
    ax1.set_ylabel(r'QSL time, $\tau_{\mathrm{QSL}}(t)$ ($\mu$s)', fontsize=14)
    ax1.set_xlim(0, 15);  ax1.set_ylim(0, 15)
    ax1.tick_params(labelsize=12)
    ax1.legend(fontsize=12)
    fig1.tight_layout()
    fig1.savefig('qsl_tau.png', dpi=300)
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # Figure 2: Time-averaged Liouvillian norm Lambda(t)
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 6.5))
    for name, (col, ls) in STYLES.items():
        ax2.plot(t_us, lam_us[name], color=col, ls=ls, lw=2.5, label=name)
    ax2.set_xlabel(r'Evolution time, $t$ ($\mu$s)', fontsize=14)
    ax2.set_ylabel(r'Time-averaged norm, $\Lambda_t$ ($\mu$s$^{-1}$)', fontsize=14)
    ax2.set_xlim(0, 15)
    ax2.tick_params(labelsize=12)
    ax2.legend(fontsize=12)
    fig2.tight_layout()
    fig2.savefig('qsl_lambda.png', dpi=300)
    plt.close(fig2)

    # -------------------------------------------------------------------------
    # Figure 3: Bures fidelity F(rho_0, rho(t))
    # -------------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(10, 6.5))
    for name, (col, ls) in STYLES.items():
        ax3.plot(t_us, fid[name], color=col, ls=ls, lw=2.5, label=name)
    ax3.set_xlabel(r'Evolution time, $t$ ($\mu$s)', fontsize=14)
    ax3.set_ylabel(r'Fidelity, $F(\hat{\rho}_0,\, \hat{\rho}(t))$', fontsize=14)
    ax3.set_xlim(0, 15);  ax3.set_ylim(0, 1.05)
    ax3.tick_params(labelsize=12)
    ax3.legend(fontsize=12)
    fig3.tight_layout()
    fig3.savefig('qsl_fidelity.png', dpi=300)
    plt.close(fig3)
