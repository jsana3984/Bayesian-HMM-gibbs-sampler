# Bayesian-HMM-gibbs-sampler
Python implementation of a custom Gibbs sampler for Bayesian parameter estimation and posterior inference in a noisy Hidden Markov Model (HMM).
# Bayesian Inference for an On/Off Communication System

## Overview
**Project Highlight:** Developed a Gibbs sampler for parameter estimation in a noisy Markov system; conducted posterior inference and simulation validation. 

This repository contains the Python code and formal mathematical report for a Bayesian inference framework applied to a hidden Markov model (HMM) representing a noisy communication device. The system alternates between two states ("Active" and "Inactive") according to a discrete-time Markov chain, but only a noisy version of the state is observed. 

The primary objective is to infer the internal dynamics of the device and the reliability of the channel using only the noisy output data. To achieve this, a custom Gibbs sampler was built from scratch to estimate the unknown transition probabilities ($\alpha$, $\beta$) and the noise parameter ($p$) from artificial data. 

## Mathematical Formulation
The hidden sequence of true states $X_n \in \{0,1\}$ is modeled as a Discrete-Time Markov Chain (DTMC). The transition matrix $P$ is defined by parameters $\alpha$ and $\beta$:

$$
P = \begin{pmatrix} 
1-\alpha & \alpha \\ 
\beta & 1-\beta 
\end{pmatrix}
$$

The observed signal $Y_n$ is subject to random bit flips (noise) due to hardware imperfections or environmental interference. This relationship is modeled as:

$$Y_n = X_n(1-I_n) + (1-X_n)I_n$$

where the error indicator is distributed as $I_n \sim Bernoulli(p)$. 

## Methodology: Gibbs Sampling
Because the posterior distribution is complex and high-dimensional, it cannot be sampled directly. Instead, the solution utilizes a Gibbs Sampler, an MCMC algorithm that samples each variable from its conditional distribution given the current values of all other variables.

**Priors:** Uninformative Uniform priors on $[0, 1]$ (equivalent to a Beta distribution) were assigned to the transition probabilities. To ensure the model is identifiable and distinguishable from a system with inverted states, the Uniform prior for the noise parameter $p$ was constrained to the interval $[0, 0.25]$.
**Conditional Updates:** Hidden states were updated via their "Markov blanket". The noise parameter $p$ was updated using rejection sampling: drawing candidates from the standard Beta distribution and rejecting any value $\ge 0.25$.

## Numerical Analysis & Results
The algorithm was implemented in Python and tested on an artificial dataset of $n=5000$ points. 

**MCMC Configuration:** The Gibbs sampler was run for 5,000 iterations, with the first 1,000 iterations discarded as a "burn-in" period to ensure the chain reached its stationary distribution.
***Parameter Recovery:** The results indicate an extremely accurate recovery of the parameters.The estimate for the noise $p$ ($0.051$) is exceptionally precise compared to the true value ($0.050$).
***Stationary Distribution:** We analyzed the stationary distribution of the system to estimate the long-run proportion of time the source is active. 

## Files in this Repository
* `ProjectStochasticSana.py`: The core Python implementation containing the data generation, Gibbs sampler algorithm, and visualization logic.
* `Stochastic_Processes_and_Simulation_project_Sana_Galil.pdf`: The comprehensive formal report detailing the conditional distributions, theoretical assumptions, and visual analysis of the MCMC posteriors.

## Tech Stack
* **Language:** Python
* **Libraries:** NumPy, SciPy, Matplotlib, Seaborn
* **Concepts:** Hidden Markov Models (HMM), Markov Chain Monte Carlo (MCMC), Gibbs Sampling, Bayesian Inference, Rejection Sampling
