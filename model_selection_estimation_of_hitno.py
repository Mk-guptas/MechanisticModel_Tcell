#estimating number of hits with combined data

def prediction_error_individual(p0, y0_initial,true_data,time1,t,model_name,nhits,m_k1):
# predicting curve based on the parameteer
    params={'k1':p0[0] ,'k3':p0[1] ,'u0':p0[2] ,'nhits':nhits ,'k2':p0[3],'d':p0[4]}
    predicted_data_1= odeint(model_name, y0_initial, t,args=(params,))
    predicted_tumor_1=np.sum(predicted_data_1[np.rint(time1).astype(int),:-1],axis=1)

    return ((predicted_tumor_1-true_data))

    
def prediction_error_combined(p0,y0_initial_1,y0_initial_2,true_data_1,true_data_2, time1,t,model_name,m_k1,nhits):
    # solving ODE using odeint for a given set of parameter ;Note: We will solve for two different initial conditions    
    params={'k1':p0[0] ,'k3':p0[1] ,'u0':p0[2] ,'nhits':nhits ,'k2':p0[3],'d':p0[4]}
    
    # predicting curve based on the parameteer
    
    predicted_data_1= odeint(model_name, y0_initial_1, t,args=(params,))
    predicted_tumor_1=np.sum(predicted_data_1[np.rint(time1).astype(int),:-1],axis=1)
    
    params['k1']=m_k1*params['k1']
    predicted_data_2= odeint(model_name, y0_initial_2, t,args=(params,))
    predicted_tumor_2=np.sum(predicted_data_2[np.rint(time1).astype(int),:-1],axis=1)

    return np.concatenate((predicted_tumor_1-true_data_1,predicted_tumor_2-true_data_2))

    
def nhit_fitting(residuals_func,model_name,nhits_list,m_k1,sheet):
    time1,dataset=reading_csv(path+'/tumor_cell_data.xlsx',sheet)
   
    for data_idx in range(2):
        residual=[];parameter=[];AIC=[] 
        for nhits_idx,nhits in enumerate(nhits_list):
            t=np.linspace(0,2900,2901)
            y0_initial=np.zeros((nhits+1)*2,);y0_initial[0]=dataset[data_idx][0] ; y0_initial[-1]=3*dataset[data_idx][0] #1st initial conditions
            residual_argument=(y0_initial,dataset[data_idx],time1,t,model_name,nhits,m_k1)
            k1=1e-4;k3=1e-2;u0=1e-3;k2=1e-3;d=1e-3
            p0 = [k1,k3,u0,k2,d]  # Initial parameter values
            bounds = ([  0,      1e-4,1e-4,1e-4,1e-3],\
                      [ 1e-4,  1e-2,1e-3,1e-2,1e-2])  
            
            res_fit = least_squares( residuals_func, p0, args=residual_argument,bounds=bounds)
            best_ssq = np.sum(res_fit.fun**2)
            log_likelihood = -(len(time1)/2)*np.log(best_ssq/len(time1)) #- (len(time1)/2)*np.log(2*np.pi) - (len(time1)/2)
            no_of_parameter=nhits*1
            AIC.append(2*(no_of_parameter+1)-2*log_likelihood)
            residual.append(best_ssq)
        
        ax[0].plot(nhits_list,residual,label=str(sheet)+str(data_idx),marker='o', linestyle='--')
        ax[1].plot(nhits_list,AIC,label=str(sheet)+str(data_idx),marker='o', linestyle='--')

        
def nhit_fitting_combined(residuals_func,model_name,nhits_list,m_k1,sheet):
    time1,dataset=reading_csv(path+'/tumor_cell_data.xlsx',sheet)
    residual=[];parameter=[] ;AIC=[]
    
    for nhits_idx,nhits in enumerate(nhits_list):
        t=np.linspace(0,2900,2901)
        y0_initial_1=np.zeros((nhits+1)*2,);y0_initial_1[0]=dataset[0][0] ; y0_initial_1[-1]=3*dataset[0][0] #1st initial conditions
        y0_initial_2=np.zeros((nhits+1)*2,);y0_initial_2[0]=dataset[1][0] ; y0_initial_2[-1]=3*dataset[1][0] #1st initial conditions
        
        residual_argument=(y0_initial_1,y0_initial_2,dataset[0],dataset[1], time1,t,model_name,m_k1,nhits)
        k1=1e-4;k3=1e-2;u0=1e-3;k2=1e-3;d=1e-3
        p0 = [k1,k3,u0,k2,d]  # Initial parameter values
        bounds = ([  0,      1e-7,1e-4,1e-6,1e-7],\
                  [ 1e-2,  1e-1,1e-3,1e-2,1e-1])  
        res_fit = least_squares( residuals_func, p0, bounds=bounds,args=residual_argument,)
        best_ssq = np.sum(res_fit.fun**2)
        if sheet =='coated':
            residual.append(best_ssq/(len(time1)))
        else:
            residual.append(best_ssq/(len(time1)))
        
        log_likelihood = -(len(time1)/2)*np.log(best_ssq/len(time1)) #- (len(time1)/2)*np.log(2*np.pi) - (len(time1)/2)
        no_of_parameter=nhits*1
        AIC.append(2*(no_of_parameter+1)-2*log_likelihood)
    ax[0].plot(nhits_list,residual,label=str(sheet),marker='o', linestyle='--')
    ax[1].plot(nhits_list,AIC,label=str(sheet),marker='o', linestyle='--')


fig,ax=plt.subplots(1,2,figsize=(10,3))
def main_inidividual():
    
    nhits_list=np.arange(2,8,1)
    
    residuals_func=prediction_error_individual
    nhit_fitting(residuals_func,model_name=model2,nhits_list=nhits_list,m_k1=1,sheet='coated')
    
    residuals_func=prediction_error_individual
    nhit_fitting(residuals_func,model_name=model2,nhits_list=nhits_list,m_k1=1,sheet='uncoated')
    
    # fitting using combined data for number of hits
    residuals_func=prediction_error_combined
    #nhit_fitting_combined(residuals_func,model_name=model2,nhits_list=nhits_list,m_k1=1,sheet='coated')
    #ax[0].set_xscale('log')
    ax[0].legend()
    plt.savefig(path2+'/nhits_fitting.svg',format="svg")

#main_inidividual()
def main_combined():
    
    nhits_list=np.arange(2,10,1)
    
    residuals_func=prediction_error_combined
    nhit_fitting_combined(residuals_func,model_name=model2,nhits_list=nhits_list,m_k1=1,sheet='coated')
    
    residuals_func=prediction_error_combined
    nhit_fitting_combined(residuals_func,model_name=model2,nhits_list=nhits_list,m_k1=0.1,sheet='uncoated')
    
    # fitting using combined data for number of hits
    residuals_func=prediction_error_combined
    #nhit_fitting_combined(residuals_func,model_name=model2,nhits_list=nhits_list,m_k1=1,sheet='coated')
    #ax[0].set_xscale('log')
    ax[0].legend()
    plt.savefig(path2+'/nhits_fitting.svg',format="svg")

main_combined()