#ESTImation of d and k2

def estimate_d_k2():
    tumor_cell_count=np.asarray([40,20,9])
    apoptosis_time=np.asarray([2,3,4])

    t_cell_count=np.asarray([21,20,13,13,6,5,8,3])
    attachment_time=np.asarray([2,4,6,8,10,11,14,15])

    d=1/(np.sum((tumor_cell_count*apoptosis_time*60))/(np.sum(tumor_cell_count)))
    k2=1/(np.sum((t_cell_count*attachment_time*60))/(np.sum(t_cell_count)))
    print('d=' ,d,'k2=',k2)
