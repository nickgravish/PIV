# %% import stuff
import numpy as np
import openpiv.tools
import openpiv.process
import openpiv.scaling
import cv2
import time
import multiprocessing 
import os
import sys
import ctypes
import sys
import scipy.io as sio
from mpi4py import MPI

nproc = MPI.COMM_WORLD.Get_size() # Size of communicator
iproc = MPI.COMM_WORLD.Get_rank() # Ranks in communicator
inode = MPI.Get_processor_name() # Node where this MPI process runs


# looks for a set of tiff ordered tiff files, starts up a pool of MPI workers to process them 


def PIVCompute(framepair, window_size = 24, overlap = 12):
    
    a, b = framepair
    frame_a = (shared_frames[a,:,:]).astype('int32')
    frame_b = (shared_frames[b,:,:]).astype('int32')
    
    tmpu, tmpv, sig2noise = openpiv.pyprocess.piv(frame_a, frame_b,
                    window_size=window_size, overlap=overlap, dt=1, 
                    sig2noise_method='peak2peak', corr_method = 'direct')
    
    x, y = openpiv.process.get_coordinates( image_size=frame_a.shape, window_size=WINDSIZE, overlap=OVERLAP )
    tmpu, tmpv, mask = openpiv.validation.sig2noise_val( tmpu, tmpv, sig2noise, threshold = 1.3)
    u, v = openpiv.filters.replace_outliers( tmpu, tmpv, method='localmean', max_iter=10, kernel_size=4)
        
    print "Waiting for " + str(framepair)
    sys.stdout.flush()

    return u, v, sig2noise


############################################################################################
# main part of code

if __name__ == '__main__':
    
    comm = MPI.COMM_WORLD 
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    path = os.getcwd()
    
    name = sys.argv[1]
    
        
# Master process
    if rank == 0:

        print "rank == 0 : " + name "\n"
        
        # parse input folder to be processed

# slave processes
    else:
        print 'rank = %d ' % MPI.COMM_WORLD.Get_rank()


MPI.Finalize()











    
    # matname = name[:-4] + str('PIV.mat')

    # if(os.path.isfile(matname) == False):
    #     # load frames
    #     frames = LoadVideo(name, startframe, NUMFRAMES)
    #     print frames.shape
    #     global shared_frames   
 
    #     shared_frames = frames
        
    #     # Prepare the u and v matrices, compute first frame
    #     start_time = time.time()
    #     x, y = openpiv.process.get_coordinates( image_size=frames.shape[1:], window_size=WINDSIZE, overlap=OVERLAP )

    #     # make the list of frame numbers to iterate over in parallel
    #     process_list = zip(range(0,frames.shape[0]), range(1,frames.shape[0]))

    #     print process_list
        
    #     # run the parallel code
    #     pool = multiprocessing.Pool(processes=PROCESSORS)
    #     result = pool.map(PIVCompute, process_list)
    #     result = np.array(result)


    #     u = np.zeros(result[:,0,:,:].shape)
    #     v = np.zeros(result[:,0,:,:].shape)

    #     # log the number of tasks executed
    #     # while (True):
    #     #     completed = result._index
    #     #     if (completed == size(process_list,0)): 
    #     #         break
            
    #     #     print "Waiting for", size(process_list,0)-completed, "tasks to complete..."
    #     #     sys.stdout.flush()
    #     #     time.sleep(2)
        

    #     # compile the results into a numpy format
    #     result = np.array(result)

    #     for kk in range(result.shape[0]):
    #         u[kk,:,:] = result[kk,0,:,:]
    #         v[kk,:,:] = result[kk,1,:,:]

    #     end_time = time.time()
    #     print repr(end_time - start_time)

    #     # save file
        
    #     sio.savemat(matname, {'u':u, 'v':v, 'x':x, 'y':y})



