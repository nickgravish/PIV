    # %% import stuff
import numpy as np
import matplotlib.pyplot as plt
import openpiv.tools
import openpiv.process
import openpiv.scaling
import cv2
import time
import scipy.io as sio
import multiprocessing 
import os
import fnmatch
import sys
import ctypes

WINDSIZE = 24
OVERLAP = 0
SEARCH = 12

PROCESSORS = 4
PERIOD = 50
WIDTH = 640
HEIGHT = 480
YWINDOW = range(0,480)

FOLDERPATH = '/Volumes/COMPATIBLE/20140822_PopupFannerKarpelson/'

global NUMFRAMES
NUMFRAMES = 500

global shared_frames
# Prepare shared memory for parallel PIV 
shared_array_base = multiprocessing.Array(ctypes.c_int32, HEIGHT*WIDTH*NUMFRAMES)
shared_frames = np.ctypeslib.as_array(shared_array_base.get_obj())
shared_frames = shared_frames.reshape((HEIGHT, WIDTH,NUMFRAMES))


def LoadVideo(name):
    print str(name)
    cap = cv2.VideoCapture(name);

    framenumber = NUMFRAMES#cap.get(7)
    height = cap.get(4)
    width = cap.get(3)
    
    # initialize video storage
    frames = np.zeros((height, width, framenumber), dtype = np.int32)
    
    #load in frames
     
    for kk in range(int(framenumber)):
        ret, tmp = cap.read(kk)
        frames[:,:,kk] = tmp[:,:,1]
        print kk
        
    return frames
    # Run PIV code, no multiprocessing
        
def ComputeMedian(frames, period):

    medframe = np.zeros(frames.shape[:2]+(period,))
    frames_adj = np.zeros(frames.shape[:2]+(frames.shape[2],))

    for kk in range(period):
        medframe[:,:,kk]  = (np.mean(frames[:,:,kk::period], axis=2)).astype(np.int32)
        
    for kk in range(frames.shape[2]):
     	frames_adj[:,:,kk] = frames[:,:,kk] - medframe[:,:,kk % period]

    return medframe, frames_adj   

def PIVCompute(args):
    
    a, b = args
    frame_a = (shared_frames[YWINDOW,:,a]).astype('int32')
    frame_b = (shared_frames[YWINDOW,:,b]).astype('int32')
    
    tmpu, tmpv, sig2noise = openpiv.pyprocess.piv(frame_a, frame_b,
                    window_size=WINDSIZE, overlap=OVERLAP, dt=1, 
                    sig2noise_method='peak2peak' )
    
    x, y = openpiv.process.get_coordinates( image_size=frame_a.shape, window_size=WINDSIZE, overlap=OVERLAP )
    tmpu, tmpv, mask = openpiv.validation.sig2noise_val( tmpu, tmpv, sig2noise, threshold = 1.3)
    u, v = openpiv.filters.replace_outliers( tmpu, tmpv, method='localmean', max_iter=10, kernel_size=4)
        
    print "Waiting for " + str(args)
    sys.stdout.flush()

    return u, v, sig2noise


	# main part of code
if __name__ == '__main__':

    # Make file list
    fnames = []
    for root, dirnames, filenames in os.walk(FOLDERPATH):
      for filename in fnmatch.filter(filenames, '*adj.avi'):
          fnames.append(os.path.join(root, filename))

    counter = 0

    for filenum in range(len(fnames)):
        name = fnames[filenum]
        matname = name[:-4] + str('PIV.mat')

        if(os.path.isfile(matname) == False):
            counter += 1
            if(counter > 10):
                exit()
                
            # load frames
            frames = LoadVideo(name)

            # Median value of frames
            medframe, frames_adj = ComputeMedian(frames,PERIOD)

            # Prepare the u and v matrices, compute first frame
            start_time = time.time()
            tmpu, tmpv, sig = PIVCompute((0,1))

            x, y = openpiv.process.get_coordinates( image_size=frames[YWINDOW,:,:].shape[:2], window_size=WINDSIZE, overlap=OVERLAP)

            u = np.zeros(tmpu.shape+(frames.shape[-1],))
            v = np.zeros(tmpv.shape+(frames.shape[-1],))	

            u[:,:,0] = tmpu
            v[:,:,0] = tmpv

            shared_frames = frames_adj

            # make the list of frame numbers to iterate over in parallel
            process_list = zip(range(1,frames.shape[-1]-1), range(2,frames.shape[-1]))

            # run the parallel code
            pool = multiprocessing.Pool(processes=PROCESSORS)
            result = pool.map(PIVCompute, process_list)

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
                u[:,:,kk+1] = result[kk,0,:,:]
                v[:,:,kk+1] = result[kk,1,:,:]

            end_time = time.time()
            print repr(end_time - start_time)

            # save file
            
            sio.savemat(matname, {'u':u, 'v':v, 'x':x, 'y':y})


