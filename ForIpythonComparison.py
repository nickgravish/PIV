    # %% import stuff
import numpy as np
import matplotlib.pyplot as plt
import openpiv.tools
import openpiv.process
import openpiv.scaling
import cv2
import time
import multiprocessing 
import os
import fnmatch
import sys
import ctypes
import sys
import scipy.io as sio


WINDSIZE = 24
OVERLAP = 12

PROCESSORS = 10

# WIDTH = 800
# HEIGHT = 400
# YWINDOW = range(0,10)

NUMFRAMES = 0

def LoadVideo(name, startframe = 0, numberframes = 0):
    """
    Loads all the frames of a video, or if framenumber is specified gives a specific starting frame
    
    !!Important, since numpy broadcasting happens from last axis to first, make the iterable framenumber the 
    first variable so that you can easily subtract off mean values from all by broadcasting

    """
    
    print str(name)
    cap = cv2.VideoCapture(name)

    # if overriding the default number of frames
    if numberframes == 0: 
        numberframes = cap.get(7)
    
    height = cap.get(4)
    width = cap.get(3)
    
    # initialize video storage
    frames = np.zeros((numberframes, height, width), dtype = np.int32)
    
    #load in frames
     
    for kk in range(int(numberframes)):
        ret, tmp = cap.read(kk+startframe)
        frames[kk,:,:] = tmp[:,:,0]
        print(str(kk) + ','),
        
    return frames



def PIVCompute(args):
    
    a, b = args
    frame_a = (shared_frames[a,:,:]).astype('int32')
    frame_b = (shared_frames[b,:,:]).astype('int32')
    
    tmpu, tmpv, sig2noise = openpiv.pyprocess.piv(frame_a, frame_b,
                    window_size=WINDSIZE, overlap=OVERLAP, dt=1, 
                    sig2noise_method='peak2peak', corr_method = 'direct')
    
    x, y = openpiv.process.get_coordinates( image_size=frame_a.shape, window_size=WINDSIZE, overlap=OVERLAP )
    tmpu, tmpv, mask = openpiv.validation.sig2noise_val( tmpu, tmpv, sig2noise, threshold = 1.3)
    u, v = openpiv.filters.replace_outliers( tmpu, tmpv, method='localmean', max_iter=10, kernel_size=4)
        
    print "Waiting for " + str(args)
    sys.stdout.flush()

    return u, v, sig2noise

    # main part of code
if __name__ == '__main__':

    path = os.getcwd()
    name = sys.argv[1]
    
    print sys.argv
 
    startframe = int(sys.argv[2])

    # name = path + '/' + name
    # print name
    
    matname = name[:-4] + str('TEST_PIV.mat')

    if(os.path.isfile(matname) == False):
        # load frames
        frames = LoadVideo(name, startframe, NUMFRAMES)
        print frames.shape
        global shared_frames
        # Prepare shared memory for parallel PIV 
        # shared_array_base = multiprocessing.Array(ctypes.c_int32, HEIGHT*WIDTH*NUMFRAMES)
        # shared_frames = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_frames = frames
        
        # Prepare the u and v matrices, compute first frame
        start_time = time.time()
        x, y = openpiv.process.get_coordinates( image_size=frames.shape[:2], window_size=WINDSIZE, overlap=OVERLAP )

        # make the list of frame numbers to iterate over in parallel
        process_list = zip(range(0,frames.shape[0]), range(1,frames.shape[0]))

        # print process_list
        
        # run the parallel code
        pool = multiprocessing.Pool(processes=PROCESSORS)
        result = pool.map(PIVCompute, process_list)
        result = np.array(result)


        u = np.zeros(result[:,0,:,:].shape)
        v = np.zeros(result[:,0,:,:].shape)

        # log the number of tasks executed
        # while (True):
        #     completed = result._index
        #     if (completed == size(process_list,0)): 
        #         break
            
        #     print "Waiting for", size(process_list,0)-completed, "tasks to complete..."
        #     sys.stdout.flush()
        #     time.sleep(2)
        

        # compile the results into a numpy format
        result = np.array(result)

        for kk in range(result.shape[0]):
            u[kk,:,:] = result[kk,0,:,:]
            v[kk,:,:] = result[kk,1,:,:]

        end_time = time.time()
        print repr(end_time - start_time)

        # save file
        
        sio.savemat(matname, {'u':u, 'v':v, 'x':x, 'y':y})


