# Quantum Effects in Magnetoreception

Master's Thesis. José Ángel Terrero Fernández. Supplementary material.

## Abstract 

The radical-pair mechanism (RPM) is the leading hypothesis for avian magnetoreception, yet the description of its open quantum dynamics remains highly debated. 
In this thesis, I study three competing master equations---Haberkorn, Jones-Hore and Kominis---used to describe spin-selective recombination within the RPM. 
By employing Monte Carlo single-molecule quantum trajectory simulations, I evaluate their internal consistency against CIDNP experimental observables. 
The numerical results indicate that the Jones-Hore formulation most accurately reproduces trajectory averages in asymmetric reaction regimes.

To assess the RPM's viability as a genuine quantum sensor, I then apply quantum information theory and formal resource theories to quantify 
singlet-triplet coherence and directional sensitivity. Quantum Fisher information analyses reveal that critical 
angular information resides within the coherence terms of the density operator, which are entirely ignored by classical product-yield measurements. 
Consequently, I propose an alternative, more optimal measurement scheme based on total magnetic moment fluctuations. 
Furthermore, I determine how measurement-based models (Jones-Hore and Kominis) impose severe penalties on coherence lifetimes and quantum speed limits. 
These frameworks restrict the operational sensing window of the compass to approximately a few microseconds but keep consistent across measures, 
whereas the Haberkorn model permits greater coherence times coming from unrealistic evolution. 
This suggests that measurement-based frameworks offer a better description of the RP's quantum dynamics.

I conclude that quantum coherence functions as a measurable and consumable resource in magnetoreception using geometric
distance-based measures such as the relative entropy of coherence.
This work establishes criteria for validating the avian compass as a genuine paradigm of quantum biology, although further experimental results are required to 
determine the exact extent to which migratory birds rely on these quantum effects.
Finally, this provides biophysical boundaries for engineering highly sensitive, bio-inspired quantum magnetometers in the future.

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
* qsl_time.py: It computes the QSL time at a fixed angle (theta) for each master equation and compares it with the unitary evolution. It also plots the
  evolution of the operator norm and the fidelity.
* quantum_speed.py: It computes the instantaneous quantum speed of the RP state at a fixed angle (theta) for each master equation.
* coherence_measures.py: It compares the time evolution of three coherence measures (l1-norm, relative entropy and geometric coherence) at a fixed angle (theta)
  for each master equations. It also plots the RPs' population decay.
