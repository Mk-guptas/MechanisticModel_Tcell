#comparison of parameters

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import norm

def comparing_parameters_with_std_computed_aysmptottically():
    # ----------------------
    # 1) Define parameters (means and std)
    # ----------------------
    n = 4  # number of replicates
    
    # --- Parameters with variability and t-test (k1, k3, μ) ---
    # Coated case:
    k1_coated    = 1.1e-5
    k3_coated    = 6.7e-3
    mu_coated    = 7.0e-4
    std_k1_coated = 2.75e-6
    std_k3_coated = 1.0e-3
    std_mu_coated = 1.0e-4
    
    # Uncoated case:
    k1_uncoated    = 0.007308120191290099
    k3_uncoated    = 0.0019081647610407677
    mu_uncoated    = 0.0005038535614962745
    std_k1_uncoated = 0.14889750795429035
    std_k3_uncoated = 0.0007181021989753726
    std_mu_uncoated = 0.000141231912471794
    
    # --- Parameters for display only (k2 and d) ---
    # Here we assume k2 and d are constant values for both cases.
    k2_coated   = 2e-3
    d_coated    = 5e-3
    
    k2_uncoated = 2e-3
    d_uncoated  = 5e-3
    
    # ----------------------
    # 2) Simulate replicate data for each parameter group
    # ----------------------
    np.random.seed(0)  # for reproducibility
    
    # For k1, k3, μ: generate random replicates based on provided std.
    data_k1_coated   = np.random.normal(k1_coated, std_k1_coated, n)
    data_k1_uncoated = np.random.normal(k1_uncoated, std_k1_uncoated, n)
    
    data_k3_coated   = np.random.normal(k3_coated, std_k3_coated, n)
    data_k3_uncoated = np.random.normal(k3_uncoated, std_k3_uncoated, n)
    
    data_mu_coated   = np.random.normal(mu_coated, std_mu_coated, n)
    data_mu_uncoated = np.random.normal(mu_uncoated, std_mu_uncoated, n)
    
    # For k2 and d: create constant arrays (no variability)
    data_k2_coated   = np.full(n, k2_coated)
    data_k2_uncoated = np.full(n, k2_uncoated)
    
    data_d_coated    = np.full(n, d_coated)
    data_d_uncoated  = np.full(n, d_uncoated)
    
    # ----------------------
    # 3) Perform statistical tests (two-sample t-test) for k1, k3, μ only
    # ----------------------
    tstat_k1, p_k1 = ttest_ind(data_k1_coated, data_k1_uncoated, equal_var=False)
    tstat_k3, p_k3 = ttest_ind(data_k3_coated, data_k3_uncoated, equal_var=False)
    tstat_mu, p_mu = ttest_ind(data_mu_coated, data_mu_uncoated, equal_var=False)
    
    print("Statistical Test Results:")
    print(f"k1 (Coated vs. Uncoated): t = {tstat_k1:.3f}, p = {p_k1:.3g}")
    print(f"k3 (Coated vs. Uncoated): t = {tstat_k3:.3f}, p = {p_k3:.3g}")
    print(f"μ  (Coated vs. Uncoated): t = {tstat_mu:.3f}, p = {p_mu:.3g}")
    
    # ----------------------
    # 4) Create a combined plot for all parameters
    # ----------------------
    fig, ax = plt.subplots(figsize=(18, 5))
    
    # Organize data for easier plotting
    param_names = ['k1', 'k3', 'μ', 'k2', 'd']
    data_dict = {
        'k1': {'Coated': data_k1_coated,   'Uncoated': data_k1_uncoated,   'p': p_k1,
               'std_Coated': std_k1_coated,  'std_Uncoated': std_k1_uncoated},
        'k3': {'Coated': data_k3_coated,   'Uncoated': data_k3_uncoated,   'p': p_k3,
               'std_Coated': std_k3_coated,  'std_Uncoated': std_k3_uncoated},
        'μ':  {'Coated': data_mu_coated,   'Uncoated': data_mu_uncoated,   'p': p_mu,
               'std_Coated': std_mu_coated,  'std_Uncoated': std_mu_uncoated},
        'k2': {'Coated': data_k2_coated,   'Uncoated': data_k2_uncoated,   'p': None,
               'std_Coated': 0,             'std_Uncoated': 0},
        'd':  {'Coated': data_d_coated,    'Uncoated': data_d_uncoated,    'p': None,
               'std_Coated': 0,             'std_Uncoated': 0},
    }
    
    # Set positions for box plots:
    # Each parameter group will have two positions (Coated and Uncoated) with a gap between groups.
    positions_coated   = []
    positions_uncoated = []
    group_centers      = []
    
    for i, name in enumerate(param_names):
        pos_coated   = i*3 + 1
        pos_uncoated = i*3 + 2
        center       = i*3 + 1.5
        positions_coated.append(pos_coated)
        positions_uncoated.append(pos_uncoated)
        group_centers.append(center)
        
        # Plot box plots for each group
        bp1 = ax.boxplot(data_dict[name]['Coated'], positions=[pos_coated], widths=0.6,
                         patch_artist=True, boxprops=dict(facecolor="lightblue"), showfliers=False)
        bp2 = ax.boxplot(data_dict[name]['Uncoated'], positions=[pos_uncoated], widths=0.6,
                         patch_artist=True, boxprops=dict(facecolor="lightgreen"), showfliers=False)
        
        # Compute means for error bar plotting
        mean_coated   = np.mean(data_dict[name]['Coated'])
        mean_uncoated = np.mean(data_dict[name]['Uncoated'])
        
        # Calculate 95% confidence intervals using 1.96 * provided std (if std available)
        ci_coated   = 1.96 * data_dict[name]['std_Coated']
        ci_uncoated = 1.96 * data_dict[name]['std_Uncoated']
        
        # Add error bars (mean ± 95% CI)
        ax.errorbar(pos_coated, mean_coated, yerr=ci_coated, fmt='o', color='red', capsize=5)
        ax.errorbar(pos_uncoated, mean_uncoated, yerr=ci_uncoated, fmt='o', color='red', capsize=5)
    
        
        # For parameters with a p-value, annotate above the box pair
        if data_dict[name]['p'] is not None:
            # Determine vertical placement for the p-value (above the higher of the two groups)
            max_val = max(np.max(data_dict[name]['Coated']), np.max(data_dict[name]['Uncoated']))
            y_p = max_val + 0.1 * abs(max_val)  # add 10% spacing
            ax.text(center, y_p, f"p = {data_dict[name]['p']:.3g}",
                    ha='center', va='bottom', fontsize=10)
    
    # Set x-ticks and labels for groups
    ax.set_xticks(group_centers)
    ax.set_xticklabels(param_names, fontsize=12)
    
    ax.set_ylabel("Parameter Value", fontsize=12)
    ax.set_title("Comparison of Parameters: Coated vs. Uncoated", fontsize=14)
    ax.set_yscale('log')
    # Set the y-axis limits as requested (from 1e-6 to 1)
    ax.set_ylim(1e-6, 1)
    
    # Create a custom legend for coated and uncoated boxes
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='lightblue', label='Coated'),
                       Patch(facecolor='lightgreen', label='Uncoated')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(path2+'/comparison.svg',format="svg")
    plt.show()

#comparing_parameters_with_std_computed_aysmptottically()


def  comparing_parameters_with_95():


    # ----------------------
    # 1)  Inputs
    # ----------------------
    # -------- k1, k3, μ  (mean and FULL 95 % width = upper – lower) --------
    
    k1_coated  = 1e-5
    lo95_k1_c  = 8e-6
    up95_k1_c  = 2e-5         # <-- now we store limits, not a width
    ci95_k1_coated=up95_k1_c-lo95_k1_c
    
    k1_uncoated  = 1e-4
    lo95_k1_u    = 1e-6
    up95_k1_u    = 1e-1
    ci95_k1_uncoated=up95_k1_u-lo95_k1_u

    
    k3_coated  = 7e-3
    lo95_k3_c  = 5e-3
    up95_k3_c  = 1e-2

    ci95_k3_coated=up95_k3_c-lo95_k3_c

    k3_uncoated  =1.8e-3
    lo95_k3_u  = 1.5e-3
    up95_k3_u  = 7e-3

    ci95_k3_uncoated=up95_k3_u-lo95_k3_u

    mu_coated  = 8e-4
    lo95_mu_c  = 6e-4
    up95_mu_c  = 1e-3


    ci95_mu_coated=up95_mu_c-lo95_mu_c

    mu_uncoated  = 5e-4
    lo95_mu_u  =2e-4
    up95_mu_u  = 6e-4

    ci95_mu_uncoated=up95_mu_u-lo95_mu_u

    
    # -------- k2, d  (mean and SD) --------
    k2_coated,  sd_k2_coated  = 2e-3, 0
    k2_uncoated,sd_k2_uncoated= 2e-3,0
    
    d_coated,   sd_d_coated   = 5e-3,0
    d_uncoated, sd_d_uncoated = 5e-3, 0
    
    # ----------------------
    # 2)  Convert CI widths to STANDARD ERRORS
    #     half-width h = (full width)/2 ;
    #     95 % CI ⇒ h = 1.96·SE  ⇒  SE = h / 1.96
    # ----------------------
    def se_from_fullwidth(lower,upper):
        return ((upper-lower) / 2.0) / 1.96
    
    se_k1_coated   = se_from_fullwidth(lo95_k1_c,up95_k1_c)
    se_k1_uncoated = se_from_fullwidth(lo95_k1_u,up95_k1_u)
    
    se_k3_coated   = se_from_fullwidth(lo95_k3_c,up95_k3_c)
    se_k3_uncoated = se_from_fullwidth(lo95_k3_u,up95_k3_u)
    
    se_mu_coated   = se_from_fullwidth(lo95_mu_c,up95_mu_c)
    se_mu_uncoated = se_from_fullwidth(lo95_mu_u,up95_mu_u)
    
    # k2, d  → treat the supplied SD as the standard error directly
    se_k2_coated   = sd_k2_coated
    se_k2_uncoated = sd_k2_uncoated
    se_d_coated    = sd_d_coated
    se_d_uncoated  = sd_d_uncoated
    
    # ----------------------
    # 3)  Large-sample z-tests (two-sided)
    # ----------------------
    def z_test(mean1, mean2, se1, se2):
        z  = (mean1 - mean2) / np.sqrt(se1**2 + se2**2)
        p  = 2 * (1 - norm.cdf(abs(z)))
        return z, p
    
    z_k1, p_k1 = z_test(k1_coated, k1_uncoated, se_k1_coated, se_k1_uncoated)
    z_k3, p_k3 = z_test(k3_coated, k3_uncoated, se_k3_coated, se_k3_uncoated)
    z_mu, p_mu = z_test(mu_coated, mu_uncoated, se_mu_coated, se_mu_uncoated)
    
    print("z-test results (large-sample normal approximation)")
    print(f"k1: z = {z_k1:.3f},  p = {p_k1:.3g}")
    print(f"k3: z = {z_k3:.3f},  p = {p_k3:.3g}")
    print(f"μ : z = {z_mu:.3f},  p = {p_mu:.3g}")
    
    # ----------------------
    # 4)  Fake replicate data JUST for box-plots
    #     (30 synthetic draws so plots look the same)
    # ----------------------
    n_display = 30
    rng = np.random.default_rng(0)
    
    def fake_draw(mean, se, n=n_display):
        return rng.normal(loc=mean, scale=se, size=n)
    
    data_dict = {
        'k1': {'Coated': fake_draw(k1_coated, se_k1_coated),
               'Uncoated': fake_draw(k1_uncoated, se_k1_uncoated),
               'p': p_k1, 'half_CI_Coated': ci95_k1_coated/2,
               'half_CI_Uncoated': ci95_k1_uncoated/2},
    
        'k3': {'Coated': fake_draw(k3_coated, se_k3_coated),
               'Uncoated': fake_draw(k3_uncoated, se_k3_uncoated),
               'p': p_k3, 'half_CI_Coated': ci95_k3_coated/2,
               'half_CI_Uncoated': ci95_k3_uncoated/2},
    
        'μ':  {'Coated': fake_draw(mu_coated, se_mu_coated),
               'Uncoated': fake_draw(mu_uncoated, se_mu_uncoated),
               'p': p_mu, 'half_CI_Coated': ci95_mu_coated/2,
               'half_CI_Uncoated': ci95_mu_uncoated/2},
    
        'k2': {'Coated': fake_draw(k2_coated, se_k2_coated),
               'Uncoated': fake_draw(k2_uncoated, se_k2_uncoated),
               'p': None, 'half_CI_Coated': 1.96*sd_k2_coated,
               'half_CI_Uncoated': 1.96*sd_k2_uncoated},
    
        'd':  {'Coated': fake_draw(d_coated,  se_d_coated),
               'Uncoated': fake_draw(d_uncoated, se_d_uncoated),
               'p': None, 'half_CI_Coated': 1.96*sd_d_coated,
               'half_CI_Uncoated': 1.96*sd_d_uncoated},
    }
    
    # ----------------------
    # 5)  Plotting (unchanged style)
    # ----------------------
    param_names = ['k1', 'k3', 'μ', 'k2', 'd']
    fig, ax = plt.subplots(figsize=(18, 5))
    
    positions_coated, positions_uncoated, group_centers = [], [], []
    for i, name in enumerate(param_names):
        pos_c, pos_u = i*3+1, i*3+2
        positions_coated.append(pos_c)
        positions_uncoated.append(pos_u)
        center = i*3 + 1.5
        group_centers.append(center)
    
        # box-plots
        ax.boxplot(data_dict[name]['Coated'],   positions=[pos_c], widths=0.6,
                   patch_artist=True, boxprops=dict(facecolor='lightblue'), showfliers=False)
        ax.boxplot(data_dict[name]['Uncoated'], positions=[pos_u], widths=0.6,
                   patch_artist=True, boxprops=dict(facecolor='lightgreen'), showfliers=False)
    
        # Mean points + CI error bars
        mean_c = np.mean(data_dict[name]['Coated'])
        mean_u = np.mean(data_dict[name]['Uncoated'])
        ax.errorbar(pos_c, mean_c, yerr=data_dict[name]['half_CI_Coated'],
                    fmt='o', color='red', capsize=5)
        ax.errorbar(pos_u, mean_u, yerr=data_dict[name]['half_CI_Uncoated'],
                    fmt='o', color='red', capsize=5)
    
        # p-value annotation (only k1, k3, μ)
        if data_dict[name]['p'] is not None:
            y_text = max(max(data_dict[name]['Coated']),
                         max(data_dict[name]['Uncoated'])) * 1.2
            ax.text(center, y_text, f"p = {data_dict[name]['p']:.3g}",
                    ha='center', va='bottom', fontsize=10)
    
    ax.set_xticks(group_centers)
    ax.set_xticklabels(param_names, fontsize=12)
    ax.set_ylabel("Parameter value", fontsize=12)
    ax.set_title("Parameters: Coated vs Uncoated (95 % CI or SD)", fontsize=14)
    ax.set_yscale('log')
    ax.set_ylim(1e-6, 1)
    
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='lightblue', label='Coated'),
                       Patch(facecolor='lightgreen', label='Uncoated')],
              loc='upper right')
    
    plt.tight_layout()
    plt.savefig(path2+'/comparison.svg', format='svg')
    plt.show()

comparing_parameters_with_95()