#COMBINED fitting and parameters calculated and plotting


## Loading the data to train the model
sheet='coated'
dfs=pd.read_excel(path+'/tumor_cell_data.xlsx', sheet_name=sheet)
exp_time=np.array(dfs['time']).flatten()*60
no_of_data=5
training_data=np.zeros((no_of_data,len(exp_time)))
for i in range(no_of_data):
    training_data[i]=np.array(dfs['dataset_'+str(i)]).flatten()
print(f'shape of the training data is {np.shape(training_data)}')



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




# Using pyhton inbuilt least square to fit the data : giving argument to fit the objective function
residuals_func=prediction_error_combined
nhits=4;model_name=model2
t = np.linspace(0, 2900, 2901)  # Time points for simulation
k1=0.00001;k3=0.01;u0=5e-4
p0 = [k1,k3,u0]  # Initial parameter values
bounds = ([  0,      1e-4,1e-4],\
          [ 1e-3,  1e-2,1e-2])  

residual_argument=(training_data,exp_time,model_name,t,nhits)

res_fit, cov_matrix, corr_matrix, std_dev,residual_var=least_square_fitting_algorithim(residuals_func,p0,bounds, residual_argument)

p_best = res_fit.x
best_ssq = np.sum(res_fit.fun**2)
print("Best-fit parameters:", p_best);print("Sum of squared residuals:", best_ssq) 
print(residual_var)
print("Standard deviations:", std_dev)
print("Correlation matrix:\n", corr_matrix)

    
### Plotting the fitted result to visulize 
if True:
    fig,ax=plt.subplots(1,2,figsize=(15,3))# plot a figure    
    if sheet=='coated':
        col1='red' ;filename='/combinedfit_treated.svg' ;col2='orange'
    else:
        col1='black' ;filename='/combinedfit_untreated.svg';col2='brown'
        
    k1,k3,u0=res_fit.x
    # Comparing the plot
    params={'k1':res_fit.x[0] ,'k3':res_fit.x[1] ,'u0':res_fit.x[2] ,'nhits':nhits }


    for data_idx,data_value in enumerate(training_data):
        
        y0_initial=np.zeros((nhits+1)*2,);y0_initial[0]=data_value[0] ; y0_initial[-1]=3*data_value[0] 
        predicted_data_1= odeint(model_name, y0_initial, t,args=(params,))
        ax[0].plot(t/60, np.sum(predicted_data_1[:,:-1],axis=1),color='black',linestyle='--')
        ax[0].scatter(exp_time/60,data_value,color=col1,label='Data 1')    
    ax[0].legend()

                   
    # Add text to display fitted parameters inside the plot
    param_text = f"k1 = {k1:.2e},{std_dev[0]:.2e}\
    \n k3 = {k3:.2e},{std_dev[1]:.2e} \n mu = {u0:.2e},{std_dev[2]:.2e}"

    ax[0].text(0, 0, param_text, fontsize=7, \
                      bbox=dict(facecolor='white', alpha=0.6),multialignment='left',transform=ax[0].transAxes)

    param_text2 = f"k1_k3 = {corr_matrix[0,1]:.2f},\
    \n k1_mu = {corr_matrix[0,2]:.2f} \n k3_mu = {corr_matrix[1,2]:.2f}"

    ax[0].text(0, 0.3, param_text2, fontsize=7, \
              bbox=dict(facecolor='white', alpha=0.6),multialignment='left',transform=ax[0].transAxes)

    
    plt.savefig(path2+filename,format="svg")
    plt.show()
    #np.save(path2+'/residual_treated_low.npy',np.asarray(residual))



    