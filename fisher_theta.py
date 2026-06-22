"""
Magnetic Sensitivity of the Radical-Pair Compass via Fisher Information
================================================================================
This script computes the quantum Fisher information (QFI) and classical Fisher
information (CFI) of the radical-pair (RP) steady state as a function of the
geomagnetic field inclination angle theta, using three different master-equation
formalisms for the spin-selective recombination kinetics:

  * Haberkorn.
  * Jones & Hore.
  * Kominis.

The QFI sets the ultimate precision limit for estimating theta via the
quantum Cramer-Rao bound: 

Delta^2(theta) >= 1 / (N * QFI), 

where N is the number of measurements. The CFI quantifies the precision achievable
with a specific observable via the standard error propagation formula.

Three CFI observables are evaluated:
  * Q_S   -- the singlet projector (related to conventional singlet yield)
  * S^2   -- total spin squared (S_1 + S_2)^2
  * S_z^2 -- square of the total z-spin (fluctuation of magnetic moment)

The steady state is obtained by integrating the master equation from t=0
to T_MAX and simultaneously accumulating its time integral, which models
the statistical mixture of RPs created continuously by optical excitation
and decaying at rate k (see the model derivation in Guo et al. 2017,
Sci. Rep. 7:5826, eq. 4).

System
------
Two unpaired electrons (donor D, acceptor A) coupled to one nuclear spin
(I = 1/2), giving an 8-dimensional Hilbert space.

Basis ordering: |uu u>, |uu d>, |ud u>, |ud d>, |du u>, |du d>, |dd u>, |dd d>
(first two indices = electron spins D and A; last = nuclear spin I).

Usage
-----
    python fisher_theta.py

Output: four PNG plots (QFI, CFI for Q_S, S^2, S_z^2).

Requirements: numpy, scipy, matplotlib, tqdm
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm


# =============================================================================
# 1. PHYSICAL PARAMETERS
# =============================================================================

GAMMA   = 8.8e4   # Electron gyromagnetic ratio  [rad s^-1 uT^-1]
B0      = 46.0    # Earth's magnetic field strength [uT]

# Axial hyperfine coupling between acceptor electron and nucleus.
# A_z >> gamma*B0 is the strong-HF regime assumed by the analytic results
# in Guo et al.; it is what drives the directional sensitivity.
A_z = 6 * GAMMA * B0    # [rad s^-1]

# Recombination rates.  kS << kT gives a long-lived singlet channel and a
# fast triplet channel, whose assymetry is relevant for avian magnetoreception.
kS = 1e4    # Singlet recombination rate  [s^-1]
kT = 1e6    # Triplet recombination rate  [s^-1]

# Integration: long enough that the RP population has fully decayed.
# The slowest timescale is 1/kS, so 15/kS is a safe upper bound.
T_MAX = 15.0 / kS


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

# Singlet and triplet projectors
s_D_dot_s_A = sDx @ sAx + sDy @ sAy + sDz @ sAz
QS = 0.25 * np.eye(8, dtype=complex) - s_D_dot_s_A
QT = np.eye(8, dtype=complex) - QS

# Observables for CFI calculations
# S^2  = (S_D + S_A)^2  -- total spin squared, related to singlet yield
# S_z^2 = (S_Dz + S_Az)^2 -- square of total z-spin, fluctuation of magnetic moment
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
# Each RP is created in the singlet electron state with the nuclear spin
# in a completely mixed state (equal probability of up and down).

state_S = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0], dtype=complex)
rho_S   = np.outer(state_S, state_S.conj())
rho_0   = np.kron(rho_S, 0.5 * I2)    # rho_e(0) x (I/2)


# =============================================================================
# 4. HAMILTONIAN
# =============================================================================

def build_hamiltonian(B0_mag, theta):
    """
    Build the spin Hamiltonian for a given field strength and inclination.

    H = gamma * B0 * [sin(theta)(S_Dx + S_Ax) + cos(theta)(S_Dz + S_Az)]
      + A_z * S_Az * I_z

    The first term is the isotropic electron Zeeman interaction. Both
    electrons couple equally to the external field because only one of them
    has a hyperfine interaction (A_x = A_y = 0, axially symmetric HF tensor),
    so it is the off-diagonal A_z term that breaks the symmetry and provides
    the directional sensitivity.

    Parameters
    ----------
    B0_mag : float -- field strength [uT]
    theta  : float -- inclination angle [rad]

    Returns
    -------
    H : ndarray, shape (8, 8)
    """
    H_zeeman    = GAMMA * B0_mag * (np.sin(theta) * (sDx + sAx) + np.cos(theta) * (sDz + sAz))
    H_hyperfine = A_z * (sAz @ Iz)
    return H_zeeman + H_hyperfine


# =============================================================================
# 5. MASTER EQUATIONS
# =============================================================================

def _pcoh(rho):
    """
    Singlet-triplet coherence measure p_coh from Kritsotakis & Kominis (2014).

    p_coh in [0, 1]:
      0 -- maximally incoherent mixture of S and T
      1 -- maximally coherent superposition

    Returns (p_coh, Tr{rho}).
    """
    tr = np.real(np.trace(rho))
    if tr < 1e-12:
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

    This is the standard spin-chemistry formalism.
    """
    drho = (  -1j * (H @ rho - rho @ H)
              - (kS / 2) * (QS @ rho + rho @ QS)
              - (kT / 2) * (QT @ rho + rho @ QT)  )
    return drho.flatten()


def L_jones_hore(rho, H):
    """
    Jones-Hore master equation:

      d rho/dt = -i[H, rho]
                 - (kS/2)(Q_S rho + rho Q_S)
                 - (kT/2)(Q_T rho + rho Q_T)
                 - (kS+kT)/2 * (rho_ST + rho_TS)

    The extra cross term damps singlet-triplet coherences without changing
    the diagonal (population) dynamics relative to Haberkorn.
    """
    rho_ST = QS @ rho @ QT
    rho_TS = QT @ rho @ QS
    drho = (  -1j * (H @ rho - rho @ H)
              - (kS / 2) * (QS @ rho + rho @ QS)
              - (kT / 2) * (QT @ rho + rho @ QT)
              - ((kS + kT) / 2) * (rho_ST + rho_TS)  )
    return drho.flatten()


def L_kominis(rho, H):
    """
    Kominis master equation:

      d rho/dt = -i[H, rho]
                 - (kS+kT)/2 * (rho_ST + rho_TS)           [dephasing]
                 - (1-p_coh)(kS rho_SS + kT rho_TT)        [incoherent recombination]
                 - (dr_S+dr_T)/Tr{rho} * (p_coh*rho_SS     [coherent recombination]
                                         + p_coh*rho_TT
                                         + rho_ST + rho_TS)

    This formalism derives the reaction terms from quantum retrodiction and
    includes explicit random singlet/triplet projections on surviving pairs.
    When the pair population drops below a numerical threshold the derivative
    is forced to zero to prevent the ODE solver from amplifying floating-point
    noise in a nearly-dead system.
    """
    p_coh, tr = _pcoh(rho)
    if tr < 1e-12:
        return np.zeros_like(rho).flatten()

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


# =============================================================================
# 6. STEADY-STATE COMPUTATION
# =============================================================================

def compute_steady_state(theta, method):
    """
    Integrate the master equation from 0 to T_MAX and return the time-integrated
    (steady-state) density matrix.

    The statistical average state of the RP ensemble under continuous optical
    excitation is:

        rho_ss = integral_0^inf  f(t) rho(t) dt,   f(t) = k exp(-k t)

    where the recombination rate k weights older RPs less. Rather than
    computing this weighted integral explicitly (which would require knowing k
    for each formalism), we instead accumulate the running time integral of
    rho(t) by augmenting the ODE state vector with a second block that simply
    integrates rho(t) -- i.e., d/dt[integral rho] = rho. This avoids a
    separate quadrature step and keeps the numerical error uniform.

    The resulting integral is then symmetrised and renormalised to compensate
    for small numerical asymmetries introduced by the stiff BDF solver.

    Parameters
    ----------
    theta  : float -- geomagnetic field inclination [rad]
    method : str   -- "Haberkorn", "Jones-Hore", or "Kominis"

    Returns
    -------
    rho_ss : ndarray, shape (8, 8) -- normalised steady-state density matrix
    """
    H = build_hamiltonian(B0, theta)

    LIOUVILLIANS = {
        "Haberkorn":  L_haberkorn,
        "Jones-Hore": L_jones_hore,
        "Kominis":    L_kominis,
    }
    L = LIOUVILLIANS[method]

    def rhs(t, y):
        rho  = y[:64].reshape((8, 8))
        drho = L(rho, H)
        # Second block accumulates the time integral of rho
        return np.concatenate([drho, y[:64]])

    y0  = np.concatenate([rho_0.flatten(), np.zeros(64, dtype=complex)])
    sol = solve_ivp(rhs, [0, T_MAX], y0, method='BDF', rtol=1e-11, atol=1e-13)

    rho_ss = sol.y[64:, -1].reshape(8, 8)

    # Symmetrise to correct small numerical Hermiticity violations
    rho_ss = 0.5 * (rho_ss + rho_ss.conj().T)
    tr = np.real(np.trace(rho_ss))
    if tr > 1e-15:
        rho_ss /= tr

    return rho_ss


# =============================================================================
# 7. FISHER INFORMATION METRICS
# =============================================================================

def compute_fisher_metrics(theta, method, dtheta=1e-4):
    """
    Compute QFI and three CFI values at a given inclination angle.

    The QFI is evaluated from the spectral decomposition of the steady state
    using the standard formula:

        QFI = 2 * sum_{i,j: p_i+p_j > 0}  |<i| drho/dtheta |j>|^2 / (p_i + p_j)

    where p_i, |i> are the eigenvalues and eigenvectors of rho_ss.

    The derivative drho/dtheta is approximated by a centred finite difference
    with step dtheta. The step must be large enough to dominate integration
    noise but small enough to keep the finite-difference error below the
    desired accuracy. dtheta=1e-4 rad is a good compromise here.

    The CFI for an observable O is given by the error-propagation formula:

        CFI(O) = (d<O>/dtheta)^2 / Var(O)

    which equals 1/Delta^2(theta) for that specific measurement scheme.

    Parameters
    ----------
    theta  : float -- evaluation point [rad]
    method : str   -- "Haberkorn", "Jones-Hore", or "Kominis"
    dtheta : float -- finite-difference step for the derivative [rad]

    Returns
    -------
    qfi     : float -- quantum Fisher information
    cfi_QS  : float -- CFI for Q_S measurement (singlet yield proxy)
    cfi_S2  : float -- CFI for S^2 measurement (total spin squared)
    cfi_Sz2 : float -- CFI for S_z^2 measurement (fluctuation of magnetic moment)
    """
    rho   = compute_steady_state(theta,          method)
    rho_p = compute_steady_state(theta + dtheta, method)
    rho_m = compute_steady_state(theta - dtheta, method)

    drho = (rho_p - rho_m) / (2 * dtheta)

    # QFI via spectral decomposition
    vals, vecs = np.linalg.eigh(rho)
    vals = np.clip(vals, 0, None)   # remove tiny negative values from numerics

    qfi = 0.0
    for i in range(8):
        for j in range(8):
            # skip terms where p_i + p_j is effectively zero to avoid divergence
            if vals[i] + vals[j] > 1e-10:
                mel  = vecs[:, i].conj() @ drho @ vecs[:, j]
                qfi += 2 * abs(mel)**2 / (vals[i] + vals[j])

    # CFI for a given observable via error propagation
    def cfi(O):
        exp_O  = np.real(np.trace(rho @ O))
        exp_O2 = np.real(np.trace(rho @ O @ O))
        var    = exp_O2 - exp_O**2
        d_exp  = np.real(np.trace(drho @ O))
        if var > 1e-11:
            return d_exp**2 / var
        return 0.0

    return float(np.real(qfi)), cfi(QS), cfi(S2_op), cfi(Sz2_op)


# =============================================================================
# 8. PLOTTING
# =============================================================================

def make_plot(theta_vals, y_H, y_JH, y_K, ylabel, filename):
    """
    Plot QFI or CFI vs theta for the three formalisms and save.

    Parameters
    ----------
    theta_vals : ndarray -- angle values [rad]
    y_H, y_JH, y_K : array-like -- Haberkorn, Jones-Hore, Kominis traces
    ylabel   : str -- y-axis label
    filename : str -- output file path
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.plot(theta_vals, y_H,  color='blue',  lw=2.5, ls='--', label='Haberkorn')
    ax.plot(theta_vals, y_JH, color='green', lw=2.5, ls='-.', label='Jones-Hore')
    ax.plot(theta_vals, y_K,  color='red',   lw=2.5,           label='Kominis')

    ax.set_xlabel(r'Direction angle, $\theta$ (rad)', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
    ax.set_xticklabels(['$0$', r'$\pi/6$', r'$\pi/4$', r'$\pi/3$', r'$\pi/2$'], fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xlim(0, np.pi/2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    ax.legend(fontsize=12)

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


# =============================================================================
# 9. MAIN
# =============================================================================

if __name__ == "__main__":

    # Sample theta uniformly from near-zero (avoids the coordinate singularity
    # at theta=0) to pi/2 (field perpendicular to the HF axis).
    theta_vals = np.linspace(1e-2, np.pi / 2, 40)

    results = {
        m: {"qfi": [], "cfi_QS": [], "cfi_S2": [], "cfi_Sz2": []}
        for m in ("Haberkorn", "Jones-Hore", "Kominis")
    }

    for method in ("Haberkorn", "Jones-Hore", "Kominis"):
        for th in tqdm(theta_vals, desc=method, ncols=80):
            qfi, cQS, cS2, cSz2 = compute_fisher_metrics(th, method)
            results[method]["qfi"].append(qfi)
            results[method]["cfi_QS"].append(cQS)
            results[method]["cfi_S2"].append(cS2)
            results[method]["cfi_Sz2"].append(cSz2)

    make_plot(
        theta_vals,
        results["Haberkorn"]["qfi"],
        results["Jones-Hore"]["qfi"],
        results["Kominis"]["qfi"],
        ylabel=r'QFI',
        filename="QFI_theta.png",
    )

    make_plot(
        theta_vals,
        results["Haberkorn"]["cfi_QS"],
        results["Jones-Hore"]["cfi_QS"],
        results["Kominis"]["cfi_QS"],
        ylabel=r'CFI ($Q_S$ measurement)  $1/\Delta^2\theta$',
        filename="CFI_QS_theta.png",
    )

    make_plot(
        theta_vals,
        results["Haberkorn"]["cfi_S2"],
        results["Jones-Hore"]["cfi_S2"],
        results["Kominis"]["cfi_S2"],
        ylabel=r'CFI ($\hat{S}^2$ measurement)  $1/\Delta^2\theta$',
        filename="CFI_S2_theta.png",
    )

    make_plot(
        theta_vals,
        results["Haberkorn"]["cfi_Sz2"],
        results["Jones-Hore"]["cfi_Sz2"],
        results["Kominis"]["cfi_Sz2"],
        ylabel=r'CFI ($\hat{S}_z^2$ measurement)  $1/\Delta^2\theta$',
        filename="CFI_Sz2_theta.png",
    )
