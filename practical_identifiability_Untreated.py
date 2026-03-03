import numpy as np
import matplotlib.pyplot as plt
from mylib import main_model,least_square_fitting_algorithim,profile_likelihood_plotter
from mylib import exp_path,sim_path
import pandas as pd
from scipy.integrate import odeint



#1. This files uses least square to estiamte the mean values and quadratic chi square assumption based variance calculation.
#2. this one is for the untreated/uncoated case only
#3. it always do the combined fittting of the data
#4. we assume that k_1 is different between cases and hence inotrduces gamma factor




## Loading the data to train the model
sheet='uncoated'
dfs=pd.read_excel(exp_path+'/tumor_cell_data.xlsx', sheet_name=sheet)
exp_time=np.array(dfs['time']).flatten()*60
no_of_data=2
training_data=np.zeros((no_of_data,len(exp_time)))
for i in range(no_of_data):
    training_data[i]=np.array(dfs['dataset_'+str(i)]).flatten()
print(f'shape of the training data is {np.shape(training_data)}')


# objective function: sum of readiual square is minimized
def prediction_error_combined(p0,training_data,exp_time,model_name,t,nhits):
    # solving ODE using odeint for a given set of parameter ;Note: We will solve for two different initial conditions    
    params={'k1':p0[0] ,'k3':p0[1] ,'u0':p0[2] ,'nhits':nhits }

    
    # predicting curve based on the parameteer
    predicted_data_set=np.zeros_like(training_data)
    for data_idx,data_value in enumerate(training_data):
        if data_idx ==1:
            params.update({"k1": p0[0]*p0[3]})
        y0_initial=np.zeros((nhits+1)*2,);y0_initial[0]=data_value[0] ; y0_initial[-1]=3*data_value[0] 
        
        predicted_data_1= odeint(model_name, y0_initial, t,args=(params,))
        predicted_data_set[data_idx]=np.sum(predicted_data_1[np.rint(exp_time).astype(int),:-1],axis=1)
        
    return predicted_data_set.flatten()-training_data.flatten()




# Using pyhton inbuilt least square to fit the data : giving argument to fit the objective function
residuals_func=prediction_error_combined
nhits=4;model_name=main_model
t = np.linspace(0, 2900, 2901)  # Time points for simulation
residual_argument=(training_data,exp_time,model_name,t,nhits)

k1=1e-6;k3=1e-2;u0=1e-3;gamma=0.1
p0 = [k1,k3,u0,gamma]  # Initial parameter values
bounds = ([  1e-8,    1e-4,1e-4,0],\
                [ 1e-1,  1e-1,1e-3,1])  
    
    
# for prolifelikelihodd
param_names = ['k1', 'k3','u0','gamma'] ;
grid_bounds = {
        'k1': (1e-6, 1e-3),
        'k3': (1e-3, 5e-2),
        'u0':(1e-4,1e-3),
        'gamma':(0,1)
    }

res_fit, cov_matrix, corr_matrix, std_dev,residual_var=profile_likelihood_plotter(residuals_func,p0,bounds, residual_argument, param_names,grid_bounds,confidence_interval=18,num_grid=25,ssq_threshold=1e3)
p_best = res_fit.x
best_ssq = np.mean(res_fit.fun**2)
print("Best-fit parameters:", p_best);print("Sum of squared residuals:", best_ssq) 
print(residual_var)
print("Standard deviations:", std_dev)
print("Correlation matrix:\n", corr_matrix)
plt.savefig(sim_path+'/profilelikelihooduntreated.svg',format="svg")

plt.show()
