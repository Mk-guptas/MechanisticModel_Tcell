#pratical Idenitifiablity analysis with 4 parameter varying gamma*k_1 between two data set
from scipy.signal import savgol_filter

def prediction_error_combined(p0,true_data_1,true_data_2, time1,t,model_name,nhits):
    # solving ODE using odeint for a given set of parameter ;Note: We will solve for two different initial conditions    
    params={'k1':p0[0] ,'k3':p0[1] ,'u0':p0[2] ,'nhits':nhits, }
    #params={'k1':p0[0] ,'k3':p0[1] ,'u0':p0[2] ,'nhits':nhits ,'d':p0[3],'k2':p0[4]}
    
    # predicting curve based on the parameteer
    y0_initial_1=np.zeros((nhits+1)*2,);y0_initial_1[0]=true_data_1[0] ; y0_initial_1[-1]=3*true_data_1[0] #1st initial conditions
    predicted_data_1= odeint(model_name, y0_initial_1, t,args=(params,))
    predicted_tumor_1=np.sum(predicted_data_1[np.rint(time1).astype(int),:-1],axis=1)
    
    params['k1']=p0[3]*p0[0]

    y0_initial_2=np.zeros((nhits+1)*2,);y0_initial_2[0]=true_data_2[0] ; y0_initial_2[-1]=3*true_data_2[0] #1st initial conditions
    predicted_data_2= odeint(model_name, y0_initial_2, t,args=(params,))
    predicted_tumor_2=np.sum(predicted_data_2[np.rint(time1).astype(int),:-1],axis=1)
    #lambdaa =1.5
    #ridge_residual = np.asarray([lambdaa*(np.dot(np.asarray(p0),np.asarray(p0)) -1)])
    return np.concatenate((predicted_tumor_1-true_data_1,predicted_tumor_2-true_data_2))
    

def main():
    model_name=model2
    nhits=4
    t = np.linspace(0, 2900, 2901)  # Time points for simulation
    
    
    
    time1,dataset=reading_csv(path+'/tumor_cell_data.xlsx','uncoated')
    
    true_data_1=dataset[0];true_data_2=dataset[1]
    
    
    # defining objective function/cost function
    residuals_func=prediction_error_combined
    residual_argument=(true_data_1,true_data_2, time1,t,model_name,nhits)
    
    
    #initial guess and bounds
    if True:
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
       
     
    print(true_data_1[0],true_data_2[0])
    
     #intial condition in the model        
    res_fit, cov_matrix, corr_matrix, std_dev,residual_var=profile_likelihood_plotter(residuals_func,p0,bounds, residual_argument, param_names,grid_bounds,confidence_interval=24.6,num_grid=10,ssq_threshold=1e3)
    
    p_best = res_fit.x
    best_ssq = np.sum(res_fit.fun**2)
    print("Best-fit parameters:", p_best);print("Sum of squared residuals:", best_ssq) 
    print(residual_var)
    print("Standard deviations:", std_dev)
    print("Correlation matrix:\n", corr_matrix)
    
    plt.savefig(path2+'/practical_idenitifiablity_2_uncoated.svg',format="svg")

main()
