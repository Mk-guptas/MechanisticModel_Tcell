def prediction_error_individual(p0, y0_initial,true_data,time1,t,model_name,nhits):
# predicting curve based on the parameteer
    params={'k1':p0[0] ,'k3':p0[1] ,'u0':p0[2] ,'nhits':nhits }
    #params={'k1':p0[0] ,'k3':p0[1] ,'u0':p0[2] ,'nhits':nhits ,'d':p0[3],'k2':p0[4]}
    predicted_data_1= odeint(model_name, y0_initial, t,args=(params,))
    predicted_tumor_1=np.sum(predicted_data_1[np.rint(time1).astype(int),:-1],axis=1)

    return ((predicted_tumor_1-true_data))


def main2():
    synthetic=True
    model_name=model2     # name the model to use
    t = np.linspace(0, 2900, 2901)  # Time points for simulation
    nhits=4
    # Generate synthetic data or Experimental data
    if synthetic==True:
        time1=np.arange(0,2500,50) # time chosen for fitting
        y0_initial=np.zeros((nhits+1)*2,);y0_initial[0]=150 ; y0_initial[-1]=3*150 #1st initial conditions
        true_params={'k1':1e-5 ,'k3':2e-3 ,'u0':5e-4 ,'nhits':nhits }
        noise_level=10
        y_true =odeint(model_name, y0_initial, t,args=(true_params,))
        predicted_tumor_1=np.sum(y_true[np.rint(time1).astype(int),:-1],axis=1)
        true_data = predicted_tumor_1 + noise_level * np.random.randn(len(predicted_tumor_1))
    else:
        time1,dataset=reading_csv(path+'/tumor_cell_data.xlsx','uncoated')
        true_data=dataset[0]
        y0_initial=np.zeros((nhits+1)*2,);y0_initial[0]=true_data[0] ; y0_initial[-1]=3*true_data[0] #1st initial conditions
        
    print(true_data[0])
    
    
    # defining objective function/cost function
    
    residuals_func=prediction_error_individual
    residual_argument=(y0_initial,true_data,time1,t,model_name,nhits)
    #initial guess and bounds
    if True:
        k1=1e-4;k3=1e-2;u0=1e-3
        p0 = [k1,k3,u0]  # Initial parameter values
        bounds = ([  0,      1e-4,1e-4],\
                  [ 5e-4,  5e-2,5e-3])  
        
        
        # for prolifelikelihodd
        param_names = ['k1', 'k3','u0'] ;
        grid_bounds = {
            'k1': (1e-6, 5e-4),
            'k3': (1e-4, 5e-2),
            'u0':(1e-4,5e-3)
        }
    
    # checking it for all 5 params
    if False:
        k1=1e-4;k3=1e-2;u0=1e-3;d=0.01;k2=0.01;p0 = [k1,k3,u0,d,k2] ;   bounds = ([  0, 1e-4,1e-4,1e-4,1e-4], [ 1e-4,  1e-2,1e-3,1e-1,1e-1])  
        # for prolifelikelihodd
        param_names = ['k1', 'k3','u0','d','k2'] ;grid_bounds = {'k1': (1e-6, 1e-4), 'k3': (1e-4, 1e-2), 'u0':(1e-4,1e-3),'d':(1e-3,1e-1), 'k2':(1e-3,1e-1)}
    
    
     #intial condition in the model        
    res_fit, cov_matrix, corr_matrix, std_dev,residual_var=profile_likelihood_plotter(residuals_func,p0,bounds, residual_argument, param_names,grid_bounds,confidence_interval=12,num_grid=10,ssq_threshold=1e3)
    
    p_best = res_fit.x
    best_ssq = np.sum(res_fit.fun**2)
    print("Best-fit parameters:", p_best);print("Sum of squared residuals:", best_ssq) 
    print(residual_var)
    print("Standard deviations:", std_dev)
    print("Correlation matrix:\n", corr_matrix)
    plt.savefig(path2+'/3_structural_idenitifiablity_1.svg',format="svg")
    #plotting the result

main2()