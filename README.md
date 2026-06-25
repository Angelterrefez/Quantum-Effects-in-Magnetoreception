# Quantum Effects in Magnetoreception

Master's Thesis. José Ángel Terrero Fernández. Supplementary material.

## Abstract 

The radical-pair mechanism (RPM) is the leading hypothesis for avian magnetoreception, yet the description of its open quantum dynamics remains highly debated. 
In this thesis, I study three competing master equations—Haberkorn, Jones-Hore, and Kominis—used to describe spin-selective recombination and the role of environmental decoherence within the RPM. 
First, by employing Monte Carlo single-molecule quantum trajectory simulations, I evaluate their internal consistency against Chemically-Induced Dynamic Nuclear Polarization (CIDNP) experimental observables. 
The numerical results indicate that the measurement-based Jones-Hore formulation most accurately reproduces ensemble trajectory averages in asymmetric reaction regimes.  

To assess the RPM's viability as a genuine quantum sensor, I then apply a quantum information theory approach. Specifically, I use statistical inference to determine the directional sensitivity of the compass 
and formal resource theories to quantify singlet-triplet coherence. Quantum Fisher Information (QFI) analyses reveal that most angular information is entirely ignored by classical product-yield measurements. 
Consequently, I propose an alternative, more optimal measurement scheme based on internal spin polarization projections. Furthermore, I determine how measurement-based models (Jones-Hore and Kominis) 
impose severe restrictions on coherence lifetimes and quantum speed limits. These frameworks restrict the operational sensing window of the compass more than the traditional Haberkorn's formulation, 
while maintaining theoretical consistency across several coherence measures. In contrast, the Haberkorn model yields longer coherence times, which arise from an inadequate treatment of quantum evolution when evaluated 
from a formal quantum information perspective. This suggests that measurement-based frameworks offer a physically more consistent description of the radical pair's open quantum dynamics.
In particular, although the Kominis framework predicts the highest theoretical sensitivity for the compass, the Jones-Hore model emerges as the most physically consistent description overall.

I conclude that quantum coherence functions as a measurable and consumable resource in magnetoreception, best quantified using geometric distance-based measures such as the relative entropy of coherence. 
This work establishes criteria for validating the avian compass as a genuine paradigm of quantum biology, although further experimental results are required to determine the exact extent to which migratory birds 
rely on these quantum effects for orientation. Finally, these findings provide fundamental biophysical boundaries for engineering highly sensitive, bio-inspired quantum magnetometers in the future.  

## Purpose of this Repository

This repository serves as the supplementary codebase for my thesis, containing all the scripts and modules used for the numerical simulations. 

**Technical Details:**
* **Language:** Python 3.12.10
* **Main Libraries:** NumPy, SciPy, Matplotlib, Pandas, Tqdm.
> **Note:** Instructions for setting up the environment and running the individual simulations follow standard python guidelines and can be found online.

## Files within this Repository

* CIDNP.py: Consistency with the quantum-trajectory picture. Comparison of radical-pair quantum dynamics in CIDNP measurements for the three master equations.
* fisher_theta.py: It computes the Fisher information (QFI and CFI for three observables) as a function of the directional angle (theta) for each master equation.
* fisher_time.py: It computes the time evolution of the Fisher information (QFI and CFI for three observables) at a fixed angle (theta) for each master equation.
* ST_coherence.py: It compares the coherence magnitudes evolution in time for the S-T and S-T_0 density matrix' terms. It uses two different Hamiltonians.
* qsl_time.py: It computes the QSL time at a fixed angle (theta) for each master equation and compares it with an unitary evolution. It also plots the
  evolution of the operator norm of the generator and the quantum fidelity.
* quantum_speed.py: It computes the instantaneous quantum speed of the RP state at a fixed angle (theta) for each master equation.
* coherence_measures.py: It compares the time evolution of three coherence measures (l1-norm, relative entropy of coherence and geometric coherence) at a fixed angle (theta)
  for each master equations. It also plots the RPs' population decay.
