def PIVCompute(frames, framenumbers, threshold = 1.3, max_iter=10, kernel_size=4, 
               WINDSIZE = 24, OVERLAP = 0, SEARCH = 12):
    """
    Perform PIV between frames 1,2
    """
    
    a, b = framenumbers
    frame_a = (frames[a,:,:]).astype('int32')
    frame_b = (frames[b,:,:]).astype('int32')
    
    tmpu, tmpv, sig2noise = openpiv.pyprocess.piv(frame_a, frame_b,
                    window_size=WINDSIZE, overlap=OVERLAP, dt=1, 
                    sig2noise_method='peak2peak', corr_method = 'direct')

#     tmpu, tmpv, sig2noise = openpiv.process.extended_search_area_piv(frame_a, frame_b,
#                     window_size=WINDSIZE, overlap=OVERLAP, dt=1, search_area_size=30,
#                     sig2noise_method='peak2peak' )
    
    x, y = openpiv.process.get_coordinates( image_size=frame_a.shape, window_size=WINDSIZE, overlap=OVERLAP )
    tmpu, tmpv, mask = openpiv.validation.sig2noise_val( tmpu, tmpv, sig2noise, threshold = 1.3)
    u, v = openpiv.filters.replace_outliers( tmpu, tmpv, method='localmean', max_iter=10, kernel_size=4)
        
#     print "Waiting for " + str(args)
#     sys.stdout.flush()

    return u, v, x, y, sig2noise
