# testing defined model work as we expected
import matplotlib.pyplot as plt
from mylib import main_model,read_excel
from scipy.integrate import odeint
import numpy as np
from mylib import exp_path


def model_test(model_type,t,params,sheet):
    fig,ax=plt.subplots(2,2, figsize=(10,6))
    time1,dataset=read_excel(exp_path+'/tumor_cell_data.xlsx',sheet)
    data_idx=0
    initial_conditions =np.zeros((nhits+1)*2,);initial_conditions[0]=dataset[data_idx][0] ; initial_conditions[-1]=3*dataset[data_idx][0]
    model_outputs = odeint(model_type, initial_conditions, t, args=(params,))
    total_population_cancer=np.sum(model_outputs[:, 0:-1],axis=1) 
    total_population_T_cell=np.sum(model_outputs[:, nhits+1:],axis=1)
    viable_population=model_outputs[:,0]+model_outputs[:,nhits+1]
    
    
    
    ax[0,0].plot(t,total_population_cancer,label=str(sheet))
    ax[0,0].plot(t,total_population_T_cell)
    ax[0,1].plot(t,viable_population/total_population_cancer,label=str(sheet))

    [axe.legend() for axe in ax.flat]
    plt.show()

nhits=3
model_type=main_model
t=np.linspace(0,2900,2901)
sheet='coated' ;
params= {'k1':1.2e-5,'k3':1e-3,'nhits':nhits}
model_test(model_type,t,params,sheet)

