import numpy as np
import tensorflow.keras.backend as K 
from skimage.measure import label
import cv2
import scipy

def getMetricsForStack(stack, mask, pixDim, errodeStruct):
    voxelVolume = pixDim[0] * pixDim[1] * pixDim[2]
    liverPixelVolume = np.count_nonzero(mask)
    liverMetricVolume = liverPixelVolume * voxelVolume
    
    mask = scipy.ndimage.binary_erosion(mask, structure=errodeStruct)

    medianHU = None
    meanHU = None
    devHU = None
    if liverPixelVolume > 0:
        values = stack[np.where(mask > 0)]
        if len(values) != 0:
            medianHU = np.median(values)
            meanHU = np.mean(values)
            devHU = np.std(values)
    return liverMetricVolume, liverPixelVolume, medianHU, meanHU, devHU
def resizeStack(stack, targetSize, method, zBatch=150):
    output = np.empty((*targetSize, stack.shape[2]))
    for zIndex in range(0, stack.shape[2], zBatch):
        maxZ = (min(zIndex+zBatch, stack.shape[2]))
        sample = stack[:,:,zIndex:maxZ]
        sample = cv2.resize(sample, (targetSize[1], targetSize[0]), interpolation=method)
        if len(sample.shape) == 2:
            sample = sample[:,:,np.newaxis]
        output[:,:,zIndex:maxZ] = sample
    return output

def thresholdImage(img, level, width, rescaleSlope=1, rescaleIntercept=0):
    img = img.astype(np.float, copy=False)
    img = img * rescaleSlope + rescaleIntercept

    img =  np.piecewise(img, 
        [img <= (level - 0.5 - (width-1)/2),
        img > (level - 0.5 + (width-1)/2)],
        [0, 255, lambda img: ((img - (level - 0.5))/(width-1) + 0.5)*(255-0)])
    img = img.astype(np.uint8, copy=False)
    return img
def getLargestCC(mask, numObjects=1):
    #mask = np.pad(mask, pad_width=1, mode='constant', constant_values=(0))
    labels = label(mask)
    binCounts = np.bincount(labels.flat)
    binCounts[0] = 0
    #print(np.argmax(binCounts))
    indsDescend = np.argsort(-binCounts)
    # Remove 0 from indices
    indsDescend = [ind for ind in indsDescend if ind != 0]
    numObjects = np.min([numObjects, len(indsDescend)])
    indsToInclude = indsDescend[0:numObjects]

    mask = np.full(labels.shape, False)
    for ind in indsToInclude:
        mask[labels == ind] = True
    #mask = mask[1:(mask.shape[0]-1), 1:(mask.shape[1]-1)]
    return mask

def reorientStack(stack, header):
    if int(header['sform_code']) == 0:
       raise ValueError("sform is not defined")

    # Change all affine values to -1, 0, or +1
    sRowX = list(header["srow_x"])[0:3]
    sRowX = np.array([1 if val > 0 else (-1 if val < 0 else 0) for val in sRowX])

    sRowY = list(header["srow_y"])[0:3]
    sRowY = np.array([1 if val > 0 else (-1 if val < 0 else 0) for val in sRowY])
    
    sRowZ = list(header["srow_z"])[0:3]
    sRowZ = np.array([1 if val > 0 else (-1 if val < 0 else 0) for val in sRowZ])
    # Ensure only a single -1 or +1 in each row of the affine matrix
    #if (np.count_nonzero(sRowX) + np.count_nonzero(sRowY) + np.count_nonzero(sRowZ)) != 3:
    #    raise ValueError("More than 1 element in a given sRow")

    # Join arrays into a single affine array
    sRow = np.array([sRowX, sRowY, sRowZ])

    # Determine if any of the stack-based axes should be flipped
    flipI = np.sum(sRow[:,0]) > 0
    flipJ = np.sum(sRow[:,1]) > 0
    flipK = np.sum(sRow[:,2]) < 0
    if flipI:
        stack = np.flip(stack, axis=0)
    if flipJ:
        stack = np.flip(stack, axis=1)
    if flipK:
        stack = np.flip(stack, axis=2)

    # Determine if any axes should be swapped
    if sRowX[0] != 0:
        swapAxes = (0,1)
    elif sRowX[2] != 0:
        swapAxes = (0,2)
    elif sRowY[2] != 0:
        swapAxes = (1,2)
    else:
        swapAxes = None

    if swapAxes != None:
        stack = np.swapaxes(stack, swapAxes[0], swapAxes[1])

    return stack

def revertStack(stack, header):
    if int(header['sform_code']) == 0:
        raise ValueError("sform is not defined")

    # Change all affine values to -1, 0, or +1
    sRowX = list(header["srow_x"])[0:3]
    sRowX = np.array([1 if val > 0 else (-1 if val < 0 else 0) for val in sRowX])

    sRowY = list(header["srow_y"])[0:3]
    sRowY = np.array([1 if val > 0 else (-1 if val < 0 else 0) for val in sRowY])
    
    sRowZ = list(header["srow_z"])[0:3]
    sRowZ = np.array([1 if val > 0 else (-1 if val < 0 else 0) for val in sRowZ])

    # Ensure only a single -1 or +1 in each row of the affine matrix
    #if (np.count_nonzero(sRowX) + np.count_nonzero(sRowY) + np.count_nonzero(sRowZ)) != 3:
    #    raise ValueError("More than 1 element in a given sRow")

    # Join arrays into a single affine array
    sRow = np.array([sRowX, sRowY, sRowZ])

    # Determine if any axes should be swapped
    if sRowX[0] != 0:
        swapAxes = (0,1)
    elif sRowX[2] != 0:
        swapAxes = (0,2)
    elif sRowY[2] != 0:
        swapAxes = (1,2)
    else:
        swapAxes = None

    if swapAxes != None:
        stack = np.swapaxes(stack, swapAxes[0], swapAxes[1])

    # Determine if any of the stack-based axes should be flipped
    flipI = np.sum(sRow[:,0]) > 0
    flipJ = np.sum(sRow[:,1]) > 0
    flipK = np.sum(sRow[:,2]) < 0
    if flipI:
        stack = np.flip(stack, axis=0)
    if flipJ:
        stack = np.flip(stack, axis=1)
    if flipK:
        stack = np.flip(stack, axis=2)
        
    return stack

def dice_coef(y_true, y_pred):
    smooth=1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)