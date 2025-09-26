import numpy as np
from scipy.integrate import odeint
from scipy.optimize import least_squares
import matplotlib.pyplot as plt



# Fisher information Method
def compute_covariance_and_correlation(res_fit,residual_func,extra_argument):
    J = res_fit.jac
    residual_variance = np.var(residual_func(res_fit.x,*extra_argument))
    cov_matrix = np.linalg.inv(J.T @ J) * residual_variance
    std_dev = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(std_dev, std_dev)
    return cov_matrix, corr_matrix, std_dev,residual_variance


# functions to compute parameter and its confidence interval 
def least_square_fitting_algorithim(residuals_func,p0,bounds, residual_argument):

    # 1. just get the parameter value and jacobian related to it
    res_fit = least_squares( residuals_func, p0, args=residual_argument,bounds=bounds)

    # 2. Covariance, standard deviation, and correlation matrices
    cov_matrix, corr_matrix, std_dev,residual_var = compute_covariance_and_correlation(res_fit,  residuals_func,residual_argument)

    return res_fit, cov_matrix, corr_matrix, std_dev,residual_var


# Function to do idenitifiablity test
def profile_likelihood_plotter(residuals_func,p0,bounds, residual_argument,param_names,grid_bounds,confidence_interval=10,num_grid=10,ssq_threshold=None):

    # 1. just get the parameter value and jacobian related to it
    res_fit = least_squares( residuals_func, p0, args=residual_argument,bounds=bounds)

    # 2. Covariance, standard deviation, and correlation matrices
    cov_matrix, corr_matrix, std_dev,residual_var = compute_covariance_and_correlation(res_fit,  residuals_func,residual_argument)


    p_best = res_fit.x  ; best_ssq = np.mean(res_fit.fun**2)  # this compute MEans squared errror
    if ssq_threshold==None:
        ssq_threshold=5*best_ssq

    # 3. Profile likelihood for each parameter

    grids = {}
    profiles = {}
    for i, name in enumerate(param_names):
        lower_bound, upper_bound = grid_bounds[name]
        grid_values, ssq_prof, params_prof = profile_likelihood_bidirectional(
            i, p_best,std_dev,bounds, lower_bound, upper_bound,  residuals_func,residual_argument,ssq_threshold,num_grid
        )
        grids[name] = grid_values
        profiles[name] = (ssq_prof, params_prof)
    
    # 4. Plot profile likelihood curves
    plot_profile_likelihoods(profiles,std_dev, p_best ,best_ssq, param_names, grids,ssq_threshold,confidence_interval)

    return res_fit, cov_matrix, corr_matrix, std_dev,residual_var

def profile_likelihood_bidirectional(param_index, p_best,std_dev,bounds, lower_bound, upper_bound, residual_func,extra_argument, ssq_threshold, num_points):
    # Create grid arrays for lower and upper directions.
    # Ensure best-fit is in the middle.

    grid_lower = np.linspace(p_best[param_index], lower_bound, num_points + 1)[1:]  # exclude best-fit
    grid_upper = np.linspace(p_best[param_index], upper_bound, num_points + 1)[:] 

    grid_values = []  # start with best-fit value
    prof_ssq = []
    prof_params = [p_best.copy()]
    
    # Helper function to compute profile likelihood at a fixed parameter value
    def compute_profile(val):
        p_fixed = p_best.copy()
        p_fixed[param_index] = val

        def res_fixed(p_free):
            p_temp = p_fixed.copy()
            free_indices = [i for i in range(len(p_best)) if i != param_index]
            for j, idx in enumerate(free_indices):
                p_temp[idx] = p_free[j]
            return residual_func(p_temp, *extra_argument)
        
        p0_free = [p_best[i] for i in range(len(p_best)) if i != param_index]
        bound_low=[bounds[0][i] for i in range(len(p_best)) if i != param_index]; bound_high=[bounds[1][i] for i in range(len(p_best)) if i != param_index]

        res_free = least_squares(res_fixed, p0_free,bounds=(bound_low,bound_high))
        current_ssq = np.mean(res_free.fun**2)
        p_opt = p_fixed.copy()
        free_indices = [i for i in range(len(p_best)) if i != param_index]
        for j, idx in enumerate(free_indices):
            p_opt[idx] = res_free.x[j]
        return current_ssq, p_opt,(bound_low,bound_high)

    
    # Scan downward (values less than best-fit)
    for val in grid_lower:  # start from closest to best-fit and move downward
        current_ssq, p_opt,fit_bound = compute_profile(val)
        if ssq_threshold is not None and current_ssq > ssq_threshold:
            break
        grid_values.insert(0, val)  # insert at beginning
        prof_ssq.insert(0, current_ssq)
        prof_params.insert(0, p_opt)
        
    print(fit_bound)
    # Scan upward (values greater than best-fit)
    for val in grid_upper:
        current_ssq, p_opt,fit_bound = compute_profile(val)
        if ssq_threshold is not None and current_ssq > ssq_threshold:
            break
        grid_values.append(val)
        prof_ssq.append(current_ssq)
        prof_params.append(p_opt)
    
    return np.array(grid_values), np.array(prof_ssq), np.array(prof_params)



def plot_profile_likelihoods(profiles,std_dev,p_best, best_ssq, param_names, grids,ssq_threshold,confidence_interval):

    n_params = len(param_names)
    plt.figure(figsize=(4 * n_params, 4))
    
    for j, name in enumerate(param_names):
        plt.subplot(1, n_params, j+1)
        ssq_prof, _ = profiles[name]
        plt.plot(grids[name][:len(ssq_prof)], ssq_prof, 'b.-', label='Profile SSQ')
        plt.axhline(best_ssq + confidence_interval, color='r', linestyle='--', label='95% line')
        plt.axhline(best_ssq, color='k', linestyle='--', label='Best fit SSQ')
        #plt.axvline(p_best[j]+std_dev[j]*1.96, color='k', linestyle='--', label='Best fit SSQ')
        #plt.axvline(p_best[j]-std_dev[j]*1.96, color='k', linestyle='--', label='Best fit SSQ')
        plt.xlabel(name)
        plt.ylabel("MSE")
        #plt.title(f"Profile likelihood for {name}")
        plt.legend()
        plt.ylim(best_ssq*0.5,ssq_threshold)
        plt.yscale('log')
        plt.xscale('log')
        #plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    
    plt.tight_layout()
    #plt.show()