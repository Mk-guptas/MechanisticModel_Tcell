#FOR OUR MODEL :Structural identifiablity analysis for our model with two data set

def prediction_error_combined(p0,y0_initial_1,y0_initial_2,true_data_1,true_data_2, time1,t,model_name,m_k1,nhits):
    # solving ODE using odeint for a given set of parameter ;Note: We will solve for two different initial conditions    
    params={'k1':p0[0] ,'k3':p0[1] ,'u0':p0[2] ,'nhits':nhits }
    #params={'k1':p0[0] ,'k3':p0[1] ,'u0':p0[2] ,'nhits':nhits ,'d':p0[3],'k2':p0[4]}
    
    # predicting curve based on the parameteer
    
    predicted_data_1= odeint(model_name, y0_initial_1, t,args=(params,))
    predicted_tumor_1=np.sum(predicted_data_1[np.rint(time1).astype(int),:-1],axis=1)
    
    params['k1']=m_k1*params['k1']
    predicted_data_2= odeint(model_name, y0_initial_2, t,args=(params,))
    predicted_tumor_2=np.sum(predicted_data_2[np.rint(time1).astype(int),:-1],axis=1)

    return np.concatenate((predicted_tumor_1-true_data_1,predicted_tumor_2-true_data_2))


def main3():
    model_name=model2
    nhits=4
    t = np.linspace(0, 2900, 2901)  # Time points for simulation
    
    synthetic=False
    if synthetic==True:
    # Generate synthetic data
        y0_initial_1=np.zeros((nhits+1)*2,);y0_initial_1[0]=150 ; y0_initial_1[-1]=3*150 #1st initial conditions
        y0_initial_2=np.zeros((nhits+1)*2,);y0_initial_2[0]=100; y0_initial_2[-1]=3*100 #1st initial conditions
        time1=np.arange(0,2500,200) # time chosen for fitting
        true_params={'k1':1e-5 ,'k3':2e-3 ,'u0':5e-4 ,'nhits':nhits }
        noise_level=10
    
        y_true =odeint(model_name, y0_initial_1, t,args=(true_params,))
        predicted_tumor_1=np.sum(y_true[np.rint(time1).astype(int),:-1],axis=1)
        true_data_1 = predicted_tumor_1 + noise_level * np.random.randn(len(predicted_tumor_1))
        y_true_2 =odeint(model_name, y0_initial_2, t,args=(true_params,))
        predicted_tumor_2=np.sum(y_true_2[np.rint(time1).astype(int),:-1],axis=1)
        true_data_2 = predicted_tumor_2 + noise_level * np.random.randn(len(predicted_tumor_2))
    else:
        time1,dataset=reading_csv(path+'/tumor_cell_data.xlsx','coated')

        true_data_1=dataset[0];true_data_2=dataset[1]
        y0_initial_1=np.zeros((nhits+1)*2,);y0_initial_1[0]=true_data_1[0] ; y0_initial_1[-1]=3*true_data_1[0] #1st initial conditions
        y0_initial_2=np.zeros((nhits+1)*2,);y0_initial_2[0]=true_data_2[0] ; y0_initial_2[-1]=3*true_data_2[0] #2nd initial conditions
        
        
    
    # defining objective function/cost function
    residuals_func=prediction_error_combined
    m_k1=1
    residual_argument=(y0_initial_1,y0_initial_2,true_data_1,true_data_2, time1,t,model_name,m_k1,nhits)
    
    
    #initial guess and bounds
    if True:
        k1=1e-4;k3=1e-2;u0=1e-3
        p0 = [k1,k3,u0]  # Initial parameter values
        bounds = ([  0,      1e-4,1e-4],\
                  [ 5e-4,  5e-2,1e-3])  
        
        
        # for prolifelikelihodd
        param_names = ['k1', 'k3','u0'] ;
        grid_bounds = {
            'k1': (1e-6, 5e-4),
            'k3': (1e-3, 5e-2),
            'u0':(1e-4,5e-3)
        }
    
    if False:
        k1=1e-4;k3=1e-2;u0=1e-3;d=0.01;k2=0.01;p0 = [k1,k3,u0,d,k2] ;   bounds = ([  0, 1e-4,1e-4,1e-4,1e-4], [ 1e-4,  1e-2,1e-3,1e-1,1e-1])  
        # for prolifelikelihodd
        param_names = ['k1', 'k3','u0','d','k2'] ;grid_bounds = {'k1': (1e-6, 1e-4), 'k3': (1e-4, 1e-2), 'u0':(1e-4,1e-3),'d':(1e-3,1e-1), 'k2':(1e-3,1e-1)}
        
           
        
    print(true_data_1[0],true_data_2[0])
    
     #intial condition in the model        
    res_fit, cov_matrix, corr_matrix, std_dev,residual_var=profile_likelihood_plotter(residuals_func,p0,bounds, residual_argument, param_names,grid_bounds,confidence_interval=18,num_grid=25,ssq_threshold=1e3)
    
        
    p_best = res_fit.x
    best_ssq = np.mean(res_fit.fun**2)
    print("Best-fit parameters:", p_best);print("Sum of squared residuals:", best_ssq) 
    print(residual_var)
    print("Standard deviations:", std_dev)
    print("Correlation matrix:\n", corr_matrix)
    plt.savefig(path2+'/3_Structural identifiablity_2_dataset.svg',format="svg")

main3()