#pearson correaltion coefficent, rank correlation coefficent, partial corelation coefficeint, partial rank correaltion coefficent, first order sobol indiece, total order sobol indices

from SALib.analyze import sobol

def main5():
    # CODE TO SIMULATE ODE MODEL FOR RANGE OF PARAMETER
    model_type = model2  # model2 should be defined as your ODE model function
    time1, dataset = reading_csv(path + '/tumor_cell_data.xlsx', 'coated')
    nhits=4  ;t=np.linspace(0,2900,2901)
    reference_data=dataset[0]  ;initial_conditions = np.zeros((nhits + 1) * 2,) ; initial_conditions[0] = reference_data[0] ;initial_conditions[-1] = 3 * reference_data[0]
    
    # Define parameter ranges for five parameters (example ranges)
    #k1 = [1e-6 ,1e-5] ;k2 = [1e-3, 1e-2] ;k3 = [1e-3, 1e-2]  ;u0 = [1e-4, 1e-3]  ;d  = [1e-3, 1e-2]   ;param_ranges = [k1, k2, k3, u0, d]  ; names= ['k1', 'k2', 'k3', 'u0', 'd'];  no_of_sample=1024
    k1 = [-5.5,-4.5] ;k2 = [-3.5, -2.5] ;k3 = [-3,-2]  ;u0 = [-4,-3]  ;d  = [-3,-2]   ;param_ranges = [k1, k2, k3, u0, d]  ; names= ['k1', 'k2', 'k3', 'u0', 'd'];  no_of_sample=1024
    #k1 = [1e-5, 1e-1] ;k2 = [1e-6, 1e-2] ;k3 = [1e-3, 1e-2]  ;mu = [0.1, 1.0]  ;d  = [1e-4, 1e-2]   ;param_ranges = [k1, k2, k3, mu, d]  ; names= ['k1', 'k2', 'k3', 'mu', 'd'];  no_of_sample=64
    problem = { 'num_vars': len(param_ranges),'names':names ,'bounds': param_ranges} 
    
    extra_parameter={'nhits':nhits}
    #extra_parameter={'k1':1e-5,'nhits':nhits}
    model_output,param_values=generating_data_ode(no_of_sample,problem,model_type,initial_conditions,t,extra_parameter)  ;print(np.shape(model_output))
    
    
    # EXTRACTING FEATURES FROM THE ODE SIMULATTION RESULT
    
    total_tumor=np.sum(model_output[:,:,:-1],axis=2)  ;print(np.shape(total_tumor))
    experimental_time_total_tumor=total_tumor[:,np.rint(time1).astype(int)]  ;print(np.shape(experimental_time_total_tumor))
    deviation_from_reference = np.asarray([np.sum((experimental_time_total_tumor[j]-reference_data)**2) for j in range(np.shape(experimental_time_total_tumor)[0])])  ;print(len(deviation_from_reference))  #1. deviation from reference data
    #deviation_from_reference = np.asarray([np.sum((experimental_time_total_tumor[j]-reference_data)**2) for j in range(np.shape(experimental_time_total_tumor)[0])])  ;print(len(deviation_from_reference))  #1. deviation from reference data
    AUC=np.asarray([np.trapz(experimental_time_total_tumor[j], np.rint(time1).astype(int)) for j in range(np.shape(experimental_time_total_tumor)[0])] ) ;print(len(AUC)) # 2. Area under the curve
     
    
    
    # CALCULATION OF DIFFERENT CORRELATION COEFFICEINT
    
    fig,ax=plt.subplots(2,2, figsize=(8,6))
    # 1. spearman rank correaltion coefficient
    spearman_deviation = {} ;
    spearman_auc = {};
    for i, name in enumerate(problem['names']):
        r_deviation, _ = spearmanr(param_values[:, i], deviation_from_reference)
        r_auc, _ = spearmanr(param_values[:, i], AUC)
        spearman_deviation[name] = r_deviation  ;spearman_auc[name] = r_auc 
    print( "Spearman correlation (deviation):", spearman_deviation)  ;print( "Spearman correlation (AUC):", spearman_auc)
    
    x = np.arange(len(names));width = 0.35  # Width of the bars
    #ax[0,0].bar(x - width/2,  [spearman_deviation[keyy] for keyy in names ], width, label='Deviation') ;ax[0,0].bar(x +width/2,  [spearman_auc[keyy] for keyy in names], width, label='AUC')
    ax[0,0].bar(names,[spearman_auc[keyy] for keyy in names],label='AUC')
    #ax[0,0].set_xticks(x) ;ax[0,0].set_xticklabels(names)
    
    
    #2. partial rank correaltion coefficient
    
    
    

    
    #3. SOBOL index
    #k1_dict={'low_lim':[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8],'upp_lim':[1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]}
    #for k1_ranges in (ki_range_dict):
    Si_1 = sobol.analyze(problem, deviation_from_reference, print_to_console=False)
    Si_2 = sobol.analyze(problem, AUC, print_to_console=False)
    print( "First order sobol indices _deviation:",  Si_1['S1'] )  ;print( "First order sobol indices _AUC:", Si_2['S1'])
    x = np.arange(len(names))
    ax[1,0].bar(x - width/2,  Si_1['S1'], width, label='Deviation') ;ax[1,0].bar(x + width/2,  Si_2['S1'], width, label='AUC')
    ax[1,0].set_xticks(x) ;ax[1,0].set_xticklabels(names) ;ax[1,0].legend()
    ax[1,1].bar(names,Si_2['S1'],label='AUC')
    
    
    plt.tight_layout()
    plt.savefig(path2+'/sensitivity analysis.svg',format="svg")
    plt.show()
    

main5()
    