
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. DATA GENERATION (The "Artificial Data")
# =======================================================
def generate_data(n, alpha_true, beta_true, p_true):
    """
    Simulates the true state X and noisy observation Y.
    """
    X = np.zeros(n, dtype=int)
    Y = np.zeros(n, dtype=int)
    
    # initial state X0 = 1 (as given in instructions)
    current_X = 1
    X[0] = current_X 
    
    # generate Markov Chain X
    for t in range(1, n):
        if current_X == 0:
            # Transition 0 -> 1 with prob alpha
            current_X = 1 if np.random.rand() < alpha_true else 0
        else:
            # Transition 1 -> 0 with prob beta
            current_X = 0 if np.random.rand() < beta_true else 1
        X[t] = current_X

    # Generate Noisy Y (Bit flip with prob p)
    for t in range(n):
        if np.random.rand() < p_true:
            Y[t] = 1 - X[t] # Flip
        else:
            Y[t] = X[t]     # Keep
            
    return X, Y

# ======================================================
# 2. GIBBS SAMPLER IMPLEMENTATION
# ==========================================
def run_gibbs_sampler(Y, n_iter, burn_in):
    n = len(Y)
    
    # --- Initialization ---
    X_sample = Y.copy() 
    
    # Initialize parameters randomly within valid ranges)
    alpha = 0.5
    beta = 0.5
    p = 0.1
    trace_alpha = np.zeros(n_iter)
    trace_beta = np.zeros(n_iter)
    trace_p = np.zeros(n_iter)
    
    #    parameters for Priors (Uniform/Beta(1,1))
    a_alpha, b_alpha = 1, 1
    a_beta, b_beta = 1, 1
    a_p, b_p = 1, 1 #    Uniform prior for p
    
    print(f"Starting Gibbs Sampler ({n_iter} iterations)...")
    
    for it in range(n_iter):
        # ---------------------------------------
        # Step 1: Sample Hidden States X 
        # --------------------------------========
        # We update each X[t] conditional on X[t-1], X[t+1], Y[t], and parameters.
        
        # here  i fixed  X[0]to 1 by the problem description that source is active at start
        X_sample[0] = 1 
        
        for t in range(1, n):
            # Calculate P(X_t = 1 )
            # The proportional probability depends on:
            #  Transition from prev state: P(X_t | X_{t-1})
            #  Emission to observation:    P(Y_t | X_t)
            # Transition to next state:   P(X_{t+1} | X_t) (only if not last state)
            
            # Log-prob for state 0 and state 1
            log_p0 = 0.0
            log_p1 = 0.0
            
            #  Transition from Previous (X_{t-1} -> X_t)
            prev = X_sample[t-1]
            if prev == 0:
                log_p0 += np.log(1 - alpha + 1e-9)
                log_p1 += np.log(alpha + 1e-9)
            else: # prev == 1
                log_p0 += np.log(beta + 1e-9)
                log_p1 += np.log(1 - beta + 1e-9)
                
            #  Emission (X_t -> Y_t)
            obs = Y[t]
            # If X=0: Match if Y=0 (prob 1-p), Mismatch if Y=1 (prob p)
            log_p0 += np.log(p + 1e-9) if obs == 1 else np.log(1 - p + 1e-9)
            # If X=1: Match if Y=1 (prob 1-p), Mismatch if Y=0 (prob p)
            log_p1 += np.log(p + 1e-9) if obs == 0 else np.log(1 - p + 1e-9)
            
            #  Transition to Next (X_t -> X_{t+1}) - Only if t < n-1
            if t < n - 1:
                next_val = X_sample[t+1]
                # If we choose X_t=0:
                # Next is next_val. Prob is alpha if next=1, (1-alpha) if next=0
                if next_val == 1:
                    log_p0 += np.log(alpha + 1e-9)
                else:
                    log_p0 += np.log(1 - alpha + 1e-9)
                
                # If we choose X_t=1:
                # Next is next_val. Prob is (1-beta) if next=1, beta if next=0
                if next_val == 1:
                    log_p1 += np.log(1 - beta + 1e-9)
                else:
                    log_p1 += np.log(beta + 1e-9)

            # Normalize and Sample
            max_log = max(log_p0, log_p1)
            prob0 = np.exp(log_p0 - max_log)
            prob1 = np.exp(log_p1 - max_log)
            prob1_norm = prob1 / (prob0 + prob1)
            
            X_sample[t] = 1 if np.random.rand() < prob1_norm else 0

        # --------------------------------------
        # Step 2: Sample Parameters given X
        # ----------------------------
        
        # Update Alpha (0 -> 1 transitions
        # Count 0->1 transitions vs 0->0 transition
        idx_0 = np.where(X_sample[:-1] == 0)[0]
        if len(idx_0) > 0:
            n_01 = np.sum(X_sample[idx_0 + 1] == 1)
            n_00 = len(idx_0) - n_01
            alpha = np.random.beta(a_alpha + n_01, b_alpha + n_00)
        else:
            #  prior if state 0 never visited
            alpha = np.random.beta(a_alpha, b_alpha)
            
        # Update Beta (1 -> 0 transitions) 
        # Count 1->0 transitions vs 1->1 transitions
        idx_1 = np.where(X_sample[:-1] == 1)[0]
        if len(idx_1) > 0:
            n_10 = np.sum(X_sample[idx_1 + 1] == 0)
            n_11 = len(idx_1) - n_10
            beta = np.random.beta(a_beta + n_10, b_beta + n_11)
        else:
            beta = np.random.beta(a_beta, b_beta)
            
        #  Update p 
        # Count mismatches between X and Y
        mismatches = np.sum(X_sample != Y)
        matches = n - mismatches
        
        # Rejection sampling Beta (p < 0.25)
        # Posterior is Beta(1 + mismatches, 1 + matches) truncated to [0, 0.25]
        while True:
            candidate_p = np.random.beta(a_p + mismatches, b_p + matches)
            if candidate_p < 0.25:
                p = candidate_p
                break
        
        #   Store
        trace_alpha[it] = alpha
        trace_beta[it] = beta
        trace_p[it] = p

    # Discard burn-in
    return trace_alpha[burn_in:], trace_beta[burn_in:], trace_p[burn_in:]

# ================
# 3. MAIN EXECUTION
# =================================
# True Parameters for Simulation ( i chhose them )
true_alpha = 0.15
true_beta = 0.20
true_p = 0.05
N_data = 5000

print(f"Generating synthetic data (n={N_data})...")
print(f"True Params: alpha={true_alpha}, beta={true_beta}, p={true_p}")
X_true, Y_obs = generate_data(N_data, true_alpha, true_beta, true_p)

# we Run Gibbs
ITERATIONS = 5000
BURN_IN = 1000
post_alpha, post_beta, post_p = run_gibbs_sampler(Y_obs, ITERATIONS, BURN_IN)

# ==========================
# 4. RESULTS & STATIONARY DISTRIBUTION

# Calculate stats
def get_stats(trace):
    mean = np.mean(trace)
    ci_lower = np.percentile(trace, 2.5)
    ci_upper = np.percentile(trace, 97.5)
    return mean, ci_lower, ci_upper

est_alpha = get_stats(post_alpha)
est_beta = get_stats(post_beta)
est_p = get_stats(post_p)

# Calculate Stationary Distribution 
# Formula: pi_1 = alpha / (alpha + beta) from the lectures
# We calculate this for EVERY sample to get a distribution
pi_active_trace = post_alpha / (post_alpha + post_beta)
est_pi = get_stats(pi_active_trace)
true_pi = true_alpha / (true_alpha + true_beta)

print("\n" + "="*40)
print("FINAL RESULTS")
print("="*40)
print(f"{'Param':<10} | {'True':<8} | {'Est Mean':<8} | {'95% CI':<15}")
print("-" * 50)
print(f"{'Alpha':<10} | {true_alpha:<8.3f} | {est_alpha[0]:<8.3f} | [{est_alpha[1]:.3f}, {est_alpha[2]:.3f}]")
print(f"{'Beta':<10} | {true_beta:<8.3f} | {est_beta[0]:<8.3f} | [{est_beta[1]:.3f}, {est_beta[2]:.3f}]")
print(f"{'p (Noise)':<10} | {true_p:<8.3f} | {est_p[0]:<8.3f} | [{est_p[1]:.3f}, {est_p[2]:.3f}]")
print("-" * 50)
print(f"{'Active %':<10} | {true_pi:<8.3f} | {est_pi[0]:<8.3f} | [{est_pi[1]:.3f}, {est_pi[2]:.3f}]")
print("="*40)

# Visualization
plt.figure(figsize=(12, 4))
plt.subplot(131); plt.hist(post_alpha, bins=30, color='skyblue', edgecolor='k'); plt.title('Posterior Alpha'); plt.axvline(true_alpha, color='r')
plt.subplot(132); plt.hist(post_beta, bins=30, color='skyblue', edgecolor='k'); plt.title('Posterior Beta'); plt.axvline(true_beta, color='r')
plt.subplot(133); plt.hist(post_p, bins=30, color='skyblue', edgecolor='k'); plt.title('Posterior p'); plt.axvline(true_p, color='r')
plt.tight_layout()
plt.show()