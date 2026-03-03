
import numpy as np
import matplotlib.pyplot as plt
from mylib import main_model,least_square_fitting_algorithim,profile_likelihood_plotter
from mylib import exp_path,sim_path
import pandas as pd
from scipy.integrate import odeint



#FOR OUR MODEL: paractical identifiablity analysis for our model with one data set generated synthetically



## generate simulation synthetic data
nhits=4
time_sampled=np.arange(0,2500,50)
t=np.linspace(0,2900,2901)


initial_conditions = np.asarray([150])
no_of_data = len(initial_conditions)

training_data = np.zeros((no_of_data, len(time_sampled)))

true_params={'k1':1e-5 ,'k3':2e-3 ,'u0':5e-4 ,'nhits':nhits }
noise_level=10

for i, init_val in enumerate(initial_conditions):
    y0 = np.zeros((nhits + 1) * 2)
    y0[0] = init_val
    y0[-1] = 3 * init_val

    y_true = odeint(main_model, y0, t, args=(true_params,))
    predicted_tumor = np.sum(y_true[time_sampled, :-1], axis=1)

    noisy_data = predicted_tumor + noise_level * np.random.randn(len(predicted_tumor))

    training_data[i, :] = noisy_data

   


# objective function: sum of readiual square is minimized
def prediction_error_combined(p0,training_data,exp_time,model_name,t,nhits):
    # solving ODE using odeint for a given set of parameter ;Note: We will solve for two different initial conditions    
    params={'k1':p0[0] ,'k3':p0[1] ,'u0':p0[2] ,'nhits':nhits }

    # predicting curve based on the parameteer
    predicted_data_set=np.zeros_like(training_data)
    for data_idx,data_value in enumerate(training_data):
        
        y0_initial=np.zeros((nhits+1)*2,);y0_initial[0]=data_value[0] ; y0_initial[-1]=3*data_value[0] 
        
        predicted_data_1= odeint(model_name, y0_initial, t,args=(params,))
        predicted_data_set[data_idx]=np.sum(predicted_data_1[np.rint(exp_time).astype(int),:-1],axis=1)
        
    return predicted_data_set.flatten()-training_data.flatten()



# defining parameters and their ranges
model_name=main_model
nhits=4
t = np.linspace(0, 2900, 2901)  # Time points for simulation


residuals_func=prediction_error_combined
residual_argument=(training_data,exp_time,model_name,t,nhits)
    
#initial guess and bounds
k1=1e-4;k3=1e-2;u0=1e-3
p0 = [k1,k3,u0]  # Initial parameter values
bounds = ([  0,      1e-4,1e-4],\
            [ 5e-4,  5e-2,1e-3])  

# bounds for plotting prolifelikelihodd
param_names = ['k1', 'k3','u0'] ;
grid_bounds = {
    'k1': (1e-6, 5e-4),
    'k3': (1e-3, 5e-2),
    'u0':(1e-4,5e-3)
    }

    
res_fit, cov_matrix, corr_matrix, std_dev,residual_var=profile_likelihood_plotter(residuals_func,p0,bounds, residual_argument, param_names,grid_bounds,confidence_interval=18,num_grid=25,ssq_threshold=1e3)
    
        
p_best = res_fit.x
best_ssq = np.mean(res_fit.fun**2)
print("Best-fit parameters:", p_best);print("Sum of squared residuals:", best_ssq) 
print(residual_var)
print("Standard deviations:", std_dev)
print("Correlation matrix:\n", corr_matrix)
plt.savefig(sim_path+'/profilelikelihoodtreated.svg',format="svg")

plt.show()
    
