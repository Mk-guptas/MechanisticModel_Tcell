
import numpy as np

def model(C, t,params):
    k1=params.get('k1',1e-5) ;k3=params.get('k3',1e-3);k2=params.get('k2',2e-3);u0=params.get('u0',5e-4);d=params.get('d',0.006);nhits=params.get('nhits',3)
    # C=[C0,C1,C2,C3 ..., C0.T,C1.T,C2.T ..., T]
    derivs=[];
    #d=0.005;k2=0.001;Kmax=500;
    Kmax=500;
    totalcell=np.sum(C[0:-1])
    v=nhits+1; # complex indexing
    for hit_idx in range(nhits+1):
        if hit_idx==0:
            derivs.append(u0*(C[hit_idx]+C[v+hit_idx])*(1-totalcell/Kmax) -(k1)*(C[hit_idx])*C[-1]+ k2*C[v+hit_idx])
        elif(hit_idx >0 and hit_idx < nhits):
            derivs.append(0*(C[hit_idx]+C[v+hit_idx])*(1-totalcell/Kmax) -(k1)*(C[hit_idx])*C[-1]\
                          +k3*C[hit_idx+v-1] +k2*C[v+hit_idx])
        elif hit_idx==nhits:
            derivs.append( k3*(C[-2])-d*C[nhits])
            
    for j in range(nhits):
        if j<nhits-1:
            derivs.append((k1)*(C[j])*C[-1]-k3*(C[v+j])-k2*C[v+j])
        elif j==nhits-1:
            derivs.append((k1)*(C[j])*C[-1]-k3*(C[v+j])-k2*C[v+j])
            
    derivs.append(-k1*C[-1]*np.sum(C[0:nhits]) +(k3+k2)*np.sum(C[nhits+1:-1]))
    return np.array(derivs)

# hit delivery does not cause the detachment
def model2(C, t,params):
    # C=[C0,C1,C2,C3 ..., C0.T,C1.T,C2.T ..., T]

    k1=params.get('k1',1e-5) ;k3=params.get('k3',1e-3);k2=params.get('k2',2e-3);u0=params.get('u0',5e-4);d=params.get('d',0.006);nhits=params.get('nhits',4)

    derivs=[];
    Kmax=500; totalcell=np.sum(C[0:-1])  #logistic growth measures
    
    v=nhits+1; # complex indexing
    
    for hit_idx in range(nhits+1):
        if hit_idx==0:
            derivs.append(u0*(C[hit_idx]+C[(nhits+1)+hit_idx])*(1-totalcell/Kmax)   -(k1)*(C[hit_idx])*C[-1]    +     k2*C[(nhits+1)+hit_idx])
        elif(hit_idx >0 and hit_idx < nhits):
            derivs.append(0*(C[hit_idx]+C[(nhits+1)+hit_idx])*(1-totalcell/Kmax)    -(k1)*(C[hit_idx])*C[-1]   +     k2*C[(nhits+1)+hit_idx])
        elif hit_idx==nhits:
            derivs.append( k3*(C[-2])-d*C[hit_idx])
            
    for complex_id in range(nhits):
        if complex_id==0:
            derivs.append(  (k1)*(C[complex_id])*C[-1]   -  k2*C[(nhits+1)+complex_id]  -   k3*C[(nhits+1)+complex_id])  
        elif complex_id<nhits:
            derivs.append(  (k1)*(C[complex_id])*C[-1]   -  k2*C[(nhits+1)+complex_id]          -k3*C[(nhits+1)+complex_id]            +k3*C[(nhits+1)+complex_id-1])

            
    derivs.append(-k1*C[-1]*np.sum(C[0:nhits]) +k2*np.sum(C[nhits+1:-1]) +k3*(C[-2]))
    
    return np.array(derivs)

# model with recovery
def model_recovery(C, t,params):

    k1=params.get('k1',1e-5) ;k3=params.get('k3',1e-3);k2=params.get('k2',2e-3);u0=params.get('u0',5e-4);d=params.get('d',0.006);nhits=params.get('nhits',3)
    r=params.get('r',0.015)
    derivs=[];

    Kmax=500;
    totalcell=np.sum(C[0:-1])
    v=nhits+1; # complex indexing
    for hit_idx in range(nhits+1):
        if hit_idx==0:
            derivs.append(u0*(C[hit_idx]+C[v+hit_idx])*(1-totalcell/Kmax) -(k1)*(C[hit_idx])*C[-1]+ k2*C[v+hit_idx] +r*C[1])
        elif(hit_idx >0 and hit_idx < nhits-1):
            derivs.append(0*(C[hit_idx]+C[v+hit_idx])*(1-totalcell/Kmax) -(k1)*(C[hit_idx])*C[-1]\
                           +k2*C[v+hit_idx]  -r*C[hit_idx]+r*C[hit_idx+1])
        elif(hit_idx >0 and hit_idx == nhits-1):
            derivs.append(0*(C[hit_idx]+C[v+hit_idx])*(1-totalcell/Kmax) -(k1)*(C[hit_idx])*C[-1]\
                           +k2*C[v+hit_idx]  -r*C[hit_idx])
        elif hit_idx==nhits:
            derivs.append( k3*(C[-2])-d*C[nhits])
            
    for j in range(nhits):
        if j==0:
            derivs.append((k1)*(C[j])*C[-1]-k2*C[v+j]-k3*C[v+j])  
        elif j<nhits:
            derivs.append((k1)*(C[j])*C[-1]-k2*C[v+j]-k3*C[v+j] +k3*C[v+j-1])

            
    derivs.append(-k1*C[-1]*np.sum(C[0:nhits]) +k2*np.sum(C[nhits+1:-1]) +k3*(C[-2]))
    return np.array(derivs)
