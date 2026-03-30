# Bayesian-HMM-gibbs-sampler
Python implementation of a custom Gibbs sampler for Bayesian parameter estimation and posterior inference in a noisy Hidden Markov Model (HMM).
# Bayesian Inference for an On/Off Communication System

## Overview
**Project Highlight:** Developed a Gibbs sampler for parameter estimation in a noisy Markov system; conducted posterior inference and simulation validation. 

[cite_start]This repository contains the Python code and formal mathematical report for a Bayesian inference framework applied to a hidden Markov model (HMM) representing a noisy communication device[cite: 6]. [cite_start]The system alternates between two states ("Active" and "Inactive") according to a discrete-time Markov chain, but only a noisy version of the state is observed[cite: 7]. 

[cite_start]The primary objective is to infer the internal dynamics of the device and the reliability of the channel using only the noisy output data[cite: 19]. [cite_start]To achieve this, a custom Gibbs sampler was built from scratch to estimate the unknown transition probabilities ($\alpha$, $\beta$) and the noise parameter ($p$) from artificial data[cite: 8]. 

## Mathematical Formulation
[cite_start]The hidden sequence of true states $X_n \in \{0,1\}$ is modeled as a Discrete-Time Markov Chain (DTMC)[cite: 27]. [cite_start]The transition matrix $P$ is defined by parameters $\alpha$ and $\beta$[cite: 30, 31]:

$$P = \begin{pmatrix} 1-\alpha & \alpha \\ \beta & 1-\beta \end{pmatrix}$$

[cite_start]The observed signal $Y_n$ is subject to random bit flips (noise) due to hardware imperfections or environmental interference[cite: 18]. [cite_start]This relationship is modeled as[cite: 37]:

$$Y_n = X_n(1-I_n) + (1-X_n)I_n$$

[cite_start]where the error indicator is distributed as $I_n \sim Bernoulli(p)$[cite: 38]. 

## Methodology: Gibbs Sampling
[cite_start]Because the posterior distribution is complex and high-dimensional, it cannot be sampled directly[cite: 48, 49]. [cite_start]Instead, the solution utilizes a Gibbs Sampler, an MCMC algorithm that samples each variable from its conditional distribution given the current values of all other variables[cite: 49].

* [cite_start]**Priors:** Uninformative Uniform priors on $[0, 1]$ (equivalent to a Beta distribution) were assigned to the transition probabilities[cite: 44]. [cite_start]To ensure the model is identifiable and distinguishable from a system with inverted states, the Uniform prior for the noise parameter $p$ was constrained to the interval $[0, 0.25]$[cite: 41, 46].
* [cite_start]**Conditional Updates:** Hidden states were updated via their "Markov blanket"[cite: 52]. [cite_start]The noise parameter $p$ was updated using rejection sampling: drawing candidates from the standard Beta distribution and rejecting any value $\ge 0.25$[cite: 66].

## Numerical Analysis & Results
[cite_start]The algorithm was implemented in Python and tested on an artificial dataset of $n=5000$ points[cite: 103]. 

* [cite_start]**MCMC Configuration:** The Gibbs sampler was run for 5,000 iterations, with the first 1,000 iterations discarded as a "burn-in" period to ensure the chain reached its stationary distribution[cite: 107].
* [cite_start]**Parameter Recovery:** The results indicate an extremely accurate recovery of the parameters[cite: 112]. [cite_start]The estimate for the noise $p$ ($0.051$) is exceptionally precise compared to the true value ($0.050$)[cite: 114].
* [cite_start]**Stationary Distribution:** We analyzed the stationary distribution of the system to estimate the long-run proportion of time the source is active[cite: 10, 168]. 

## Files in this Repository
* `ProjectStochasticSana.py`: The core Python implementation containing the data generation, Gibbs sampler algorithm, and visualization logic.
* `Stochastic_Processes_and_Simulation_project_Sana_Galil.pdf`: The comprehensive formal report detailing the conditional distributions, theoretical assumptions, and visual analysis of the MCMC posteriors.

## Tech Stack
* **Language:** Python
* **Libraries:** NumPy, SciPy, Matplotlib, Seaborn
* **Concepts:** Hidden Markov Models (HMM), Markov Chain Monte Carlo (MCMC), Gibbs Sampling, Bayesian Inference, Rejection Sampling
