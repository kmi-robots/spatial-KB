import numpy as np

area_labels = ['XS','small','medium','large','XL']
depth_labels = ['flat','thin','thick','bulky']

def quantize(estimated_dims, lam, T, crop_shape):
    depth = min(estimated_dims)  # bc KB is for all three configurations of d1,d2,d3 here we make the assumption of considering only one configuration
    estimated_dims.remove(depth)  # i.e., the one where the min of the three is taken as depth
    d1, d2 = estimated_dims

    """ size quantization: from quantitative dims to qualitative labels"""
    qual = quant_size_qual(d1, d2, thresholds=T)
    flat = quant_flat(depth, len_thresh=lam[0])
    flat_flag = 'flat' if flat else 'non flat'
    # Aspect ratio based on crop
    aspect_ratio = quant_AR(crop_shape, (d1, d2))
    thinness = quant_thinness(depth, cuts=lam)
    # cluster = qual + "-" + thinness

    # print("Detected size is %s" % qual)
    # print("Object is %s" % flat_flag)
    # print("Object is %s" % aspect_ratio)
    # print("Object is %s" % thinness)
    return qual,flat,aspect_ratio,thinness

def quant_size_qual(dim1, dim2,thresholds=[]): #): #t3=0.19

    estimated_area = np.log(dim1 * dim2)
    if estimated_area < thresholds[0]: return 'XS'
    elif estimated_area >= thresholds[-1]: return 'XL'
    else: #intermediate cases
        for i in range(len(thresholds)-1):
            if (estimated_area>=thresholds[i] and estimated_area < thresholds[i+1]):
                return area_labels[i+1]

def quant_flat(depth, len_thresh = 0.0): #if depth greater than x% of its min dim then non flat
    depth = np.log(depth)
    if depth <= len_thresh: return True
    else: return False

def quant_thinness(depth, cuts=[]):
    """
    Rates object thinness/thickness based on measured depth
    """
    depth = np.log(depth)
    if depth <= cuts[0]: return 'flat'
    elif depth > cuts[-1]: return 'bulky'
    else: # intermediate cases
        for i in range(len(cuts)-1):
            if depth > cuts[i] and depth <= cuts[i+1]:
                return depth_labels[i+1]

def quant_AR(crop_dims,estim_dims, t=1.4):
    """
    Returns aspect ration based on 2D crop dimensions
    and estimated dimensions
    """
    height, width = crop_dims #used to derive the orientation
    # print("crop dimensions are %s x %s" % (str(width), str(height)))
    if height >= width:
        #h = max(d1,d2)
        #w = min(d1,d2)
        AR = height/width
        if AR >= t: return 'TTW'
        else: return 'EQ' #h and w are comparable
    if height < width:
        #h = min(d1, d2)
        #w = max(d1, d2)
        AR = width/height
        if AR >= t: return 'WTT'
        else: return 'EQ' #h and w are comparable

