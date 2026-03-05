from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from  scipy.optimize import least_squares
from mylib import main_model,read_excel
from mylib import exp_path,sim_path
import pandas as pd


# 1. this file fit the experimental data and idenitfy the no of hits


### Loading the data to train the model  ################################
sheet='coated'
dfs=pd.read_excel(exp_path+'/tumor_cell_data.xlsx', sheet_name=sheet)
exp_time=np.array(dfs['time']).flatten()*60
no_of_data=2
training_data_treated=np.zeros((no_of_data,len(exp_time)))
for i in range(no_of_data):
    training_data_treated[i]=np.array(dfs['dataset_'+str(i)]).flatten()
print(f'shape of the training data is {np.shape(training_data_treated)}')

sheet='uncoated'
dfs=pd.read_excel(exp_path+'/tumor_cell_data.xlsx', sheet_name=sheet)
exp_time=np.array(dfs['time']).flatten()*60
no_of_data=2
training_data_untreated=np.zeros((no_of_data,len(exp_time)))
for i in range(no_of_data):
    training_data_untreated[i]=np.array(dfs['dataset_'+str(i)]).flatten()
print(f'shape of the training data is {np.shape(training_data_untreated)}')



# objective function: sum of readiual square is minimized           ########################################################
def prediction_error_combined(p0,training_data,exp_time,model_name,t,nhits,gamma):
    # solving ODE using odeint for a given set of parameter ;Note: We will solve for two different initial conditions    
    params={'k1':p0[0] ,'k3':p0[1] ,'u0':p0[2] ,'nhits':nhits,'k2':p0[3],'d':p0[4] }

    
    # predicting curve based on the parameteer
    predicted_data_set=np.zeros_like(training_data)
    for data_idx,data_value in enumerate(training_data):
        if data_idx ==1:
            params.update({"k1": p0[0]*gamma})

        y0_initial=np.zeros((nhits+1)*2,);y0_initial[0]=data_value[0] ; y0_initial[-1]=3*data_value[0] 
        
        predicted_data_1= odeint(model_name, y0_initial, t,args=(params,))
        predicted_data_set[data_idx]=np.sum(predicted_data_1[np.rint(exp_time).astype(int),:-1],axis=1)
        
    return predicted_data_set.flatten()-training_data.flatten()


# parameters 
nhits_list=np.arange(2,10,1)      

residuals_func=prediction_error_combined
t=np.linspace(0,2900,2901)

residual_treated=[] ;AIC_treated=[]; 
residual_untreated=[];AIC_untreated=[];

fig,ax=plt.subplots(1,2,figsize=(10,3))

for nhits_idx,nhits in enumerate(nhits_list):

    k1=1e-4;k3=1e-2;u0=1e-3;k2=1e-3;d=1e-3
    p0 = [k1,k3,u0,k2,d]  # Initial parameter values
    bounds = ([  0,      1e-5,1e-5,1e-4,1e-4],\
                [ 1e-2,  1e-1,1e-2,1e-2,1e-1])  
    
    # first for the treated 
    residual_argument=(training_data_treated,exp_time,main_model,t,nhits,1)
    
    res_fit = least_squares( residuals_func, p0, bounds=bounds,args=residual_argument,)

    best_ssq = np.sum(res_fit.fun**2)
    residual_treated.append(best_ssq/(len(exp_time)))
    
    log_likelihood = -(len(exp_time)/2)*np.log(best_ssq/len(exp_time)) #- (len(time1)/2)*np.log(2*np.pi) - (len(time1)/2)
    no_of_parameter=nhits*1
    AIC_treated.append(2*(no_of_parameter+1)-2*log_likelihood)

    #now for the untreated
    residual_argument=(training_data_untreated,exp_time,main_model,t,nhits,0.1)
    
    res_fit = least_squares( residuals_func, p0, bounds=bounds,args=residual_argument,)

    best_ssq = np.sum(res_fit.fun**2)
    residual_untreated.append(best_ssq/(len(exp_time)))
    
    log_likelihood = -(len(exp_time)/2)*np.log(best_ssq/len(exp_time)) #- (len(time1)/2)*np.log(2*np.pi) - (len(time1)/2)
    no_of_parameter=nhits*1
    AIC_untreated.append(2*(no_of_parameter+1)-2*log_likelihood)


ax[0].plot(nhits_list,residual_treated,label='treated',marker='o', linestyle='--',)
ax[0].plot(nhits_list,residual_untreated,marker='o', linestyle='--',label='untreated')
ax[0].legend()

ax[1].plot(nhits_list,AIC_treated,marker='o', linestyle='--',label='treated')
ax[1].plot(nhits_list,AIC_untreated,marker='o', linestyle='--',label='untreated')
ax[1].legend()

plt.show()
plt.savefig(sim_path+'/nhits_fitting.svg',format="svg")
