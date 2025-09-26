# testing defined model work as we expected


def model_test(model_type,t,p0,sheet):
    time1,dataset=reading_csv(path+'/tumor_cell_data.xlsx',sheet)
    data_idx=0
    initial_conditions =np.zeros((nhits+1)*2,);initial_conditions[0]=dataset[data_idx][0] ; initial_conditions[-1]=3*dataset[data_idx][0]
    model_outputs = odeint(model_type, initial_conditions, t, args=(*p0,))
    total_population_cancer=np.sum(model_outputs[:, 0:-1],axis=1) 
    total_population_T_cell=np.sum(model_outputs[:, nhits+1:],axis=1)
    viable_population=model_outputs[:,0]+model_outputs[:,nhits+1]
    
    
    
    ax[0,0].plot(t,total_population_cancer,label=str(sheet))
    ax[0,0].plot(t,total_population_T_cell)
    ax[0,1].plot(t,viable_population/total_population_cancer,label=str(sheet))
    ax[1,0].plot(t,p0[-1]*(viable_population/total_population_cancer),label=str(sheet))
    ax[0,0].legend();ax[0,1].legend();ax[1,0].legend()

    nhits=3
    model_type=model2
    t=np.linspace(0,2900,2901)
    sheet='coated' ;p0= [1.22e-5,5.22e-3,8.16e-4]
    
    fig,ax=plt.subplots(2,2, figsize=(8,4))
    plotter_effective(model_type,t,p0,sheet)
    sheet='uncoated';p0=[5e-5,1.5e-3,4.7e-4]
    plotter_effective(model_type,t,p0,sheet)