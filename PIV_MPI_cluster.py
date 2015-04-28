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
from PIL import Image
import pypar as pp

num_processors = pp.size()
rank = pp.rank()
node = pp.get_processor_name()

# MPI Constants
MASTER_PROCESS = 0
WORK_TAG = 1
DIE_TAG = 2

DeltaT = 1 # frames to skip
window_size = 24
overlap = 12



def PIVCompute(frame_a, frame_b, window_size = 24, overlap = 12):
    
    tmpu, tmpv, sig2noise = openpiv.pyprocess.piv(frame_a, frame_b,
                    window_size=window_size, overlap=overlap, dt=1, 
                    sig2noise_method='peak2peak', corr_method = 'direct')
    
    x, y = openpiv.process.get_coordinates( image_size=frame_a.shape, window_size=window_size, overlap=overlap)
    tmpu, tmpv, mask = openpiv.validation.sig2noise_val( tmpu, tmpv, sig2noise, threshold = 1.3)
    u, v = openpiv.filters.replace_outliers( tmpu, tmpv, method='localmean', max_iter=10, kernel_size=4)
    
    return u, v


############################################################################################
# main part of code

if __name__ == '__main__':
    
    vidpath = sys.argv[1]
    foldername = os.getcwd().split('/')[-1]

    tif_files = [f for f in os.listdir(vidpath) if f.endswith('.tif')]

# !!!! DEBUG
    tif_files = tif_files[:10]


# Create list of file pairs to process
    process_list = zip(range(0,len(tif_files)), range(DeltaT,len(tif_files)-DeltaT))
    work_size = len(process_list)
    
    # Dispatch jobs to worker processes
    work_index = 0
    num_completed = 0

    # Master process
    if rank == MASTER_PROCESS:

        frame_a = np.array(Image.open(os.path.join(vidpath, tif_files[0])));
        frame_b = np.array(Image.open(os.path.join(vidpath, tif_files[1])));

        # run PIV computation to get size of matrix and x,y's
        x, y = openpiv.process.get_coordinates(image_size=frame_a.shape,
                                                 window_size=window_size, overlap=overlap)

        # pre-allocate the u, v matrices
        u = np.zeros((work_size, x.shape[0], x.shape[1]))
        v = np.zeros((work_size, x.shape[0], x.shape[1]))

        # Start all worker processes
        for i in range(1, min(num_processors, work_size)):
            pp.send(work_index, i, tag=WORK_TAG)
            pp.send(process_list[work_index], i)
            print "Sent process list " + str(process_list[i]) + " to processor " + str(i)
            work_index += 1

        # Receive results from each worker, and send it new data
        for i in range(num_processors, work_size):
            results, status = pp.receive(source=pp.any_source, tag=pp.any_tag, return_status=True)
            index = status.tag

            # receive and parse the resulting var
            u[index,:,:] = results[0,:,:]
            v[index,:,:] = results[1,:,:]

            proc = status.source
            num_completed += 1
            work_index += 1
            pp.send(work_index, proc, tag=WORK_TAG)
            pp.send(process_list[work_index], proc)
            print "Sent work index " + str(work_index) + " to processor " + str(proc)

        # Get results from remaining worker processes
        while num_completed < work_size-1:
            results, status = pp.receive(source=pp.any_source, tag=pp.any_tag, return_status=True)
            index = status.tag
            
            # receive and parse the resulting var
            u[index,:,:] = results[0,:,:]
            v[index,:,:] = results[1,:,:]

            num_completed += 1

        # Shut down worker processes
        for proc in range(1, num_processors):
            print "Stopping worker process " + str(proc)
            pp.send(-1, proc, tag=DIE_TAG)

        # Package up the results to save, also save all the PIV parameters
        sio.savemat(os.path.join(os.getcwd(), '../' + 'test.mat'),{'x':x, 'y':y, 'u':u, 'v':v, 
                                                                    'window_size':window_size,
                                                                    'overlap':overlap})

    else:
        ### Worker Processes ###
        continue_working = True
        while continue_working:

            # check if being put to sleep
            work_index, status =  pp.receive(source=MASTER_PROCESS, tag=pp.any_tag, 
                    return_status=True)

            if status.tag == DIE_TAG:
                continue_working = False
            
            # not being put to sleep, load in videos of interest and compute
            else:
                frame_pair, status = pp.receive(source=MASTER_PROCESS, tag=pp.any_tag, 
                    return_status=True)
                work_index = status.tag
                
                print  "Received work frame pair " + str(frame_pair)                 

                frame_a = np.array(Image.open(os.path.join(vidpath, tif_files[frame_pair[0]])));
                frame_b = np.array(Image.open(os.path.join(vidpath, tif_files[frame_pair[1]])));
                
                # Code below simulates a task running
                u, v = PIVCompute(frame_a, frame_b, window_size = window_size, overlap = overlap)
                
                # package up into work array
                work_array = np.zeros((2,u.shape[0], u.shape[1]))
                work_array[0,:,:] = u
                work_array[1,:,:] = v

                result_array = work_array.copy()

                pp.send(result_array, destination=MASTER_PROCESS, tag=work_index)
        #### while
    #### if worker

    pp.finalize()












    
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



