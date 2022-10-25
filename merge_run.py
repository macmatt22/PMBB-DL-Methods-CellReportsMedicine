#
# Output is written to data/liver.mask.nii.gz and data/spleen.mask.nii.gz
#
import tensorflow
import utils
import numpy as np
import nibabel as nib
import cv2



modelLiverFilePath = "/home/mattmac22/projects/net_learn/liver_and_spleen/merge_run_old/weights/liver_weights.h5"
modelSpleenFilePath = "/home/mattmac22/projects/net_learn/liver_and_spleen/merge_run_old/weights/spleen_weights.h5"
modelContrastFilePath = "/home/mattmac22/projects/net_learn/liver_and_spleen/merge_run_old/weights/contrast_weights.h5"
stackFilePath = "/home/mattmac22/projects/net_learn/liver_and_spleen/merge_run_old/data/test_stack.nii.gz"

outputLiverMaskFilePath = "/home/mattmac22/projects/net_learn/liver_and_spleen/merge_run_old/data/liver.mask.nii.gz"
outputSpleenMaskFilePath = "/home/mattmac22/projects/net_learn/liver_and_spleen/merge_run_old/data/spleen.mask.nii.gz"

zBatchSize = 32

modelLiver = tensorflow.keras.models.load_model(modelLiverFilePath, custom_objects={"dice_coef": utils.dice_coef})
modelSpleen = tensorflow.keras.models.load_model(modelSpleenFilePath, custom_objects={"dice_coef": utils.dice_coef})
modelContrast = tensorflow.keras.models.load_model(modelContrastFilePath)

imgShape = (modelLiver.input.shape[1], modelLiver.input.shape[2], modelLiver.input.shape[3])

niiInput = nib.load(stackFilePath)
stack = niiInput.get_fdata()
			
stack = utils.reorientStack(stack, niiInput.header)
stackOriginal = stack.copy()
originalShape = stack.shape[0:2]
stack = utils.resizeStack(stack, imgShape[0:2], method=cv2.INTER_CUBIC)
stack = utils.thresholdImage(stack, level=30, width=150)
stack = np.moveaxis(stack, -1, 0)

maskLiver = np.empty(stack.shape, dtype=np.float)
maskSpleen = np.empty(stack.shape, dtype=np.float)

stack = stack[:,:,:,np.newaxis]

chanceContrast = 0.0

for zIndex in range(0, stack.shape[0], zBatchSize):
    print(zIndex)
    maxZ = (min(zIndex+zBatchSize, stack.shape[0])) 
    sample = stack[zIndex:maxZ, :, :, :]

    predictionList = modelContrast.predict(sample)
    chanceContrast += np.sum(predictionList[:,0])
chanceContrast = chanceContrast / stack.shape[0]
print("Chance scan contains IV-contrast: %0.2f" % chanceContrast)

if chanceContrast >= 0.50:
    pass

else:
    for zIndex in range(0, stack.shape[0], zBatchSize):
        print(zIndex)
        maxZ = (min(zIndex+zBatchSize, stack.shape[0])) 
        sample = stack[zIndex:maxZ, :, :, :]
        
        # Liver
        samplePredLiver = modelLiver.predict(sample)
        samplePredLiver = samplePredLiver[:,:,:,0]
        samplePredLiver[samplePredLiver >= 0.50] = 1
        samplePredLiver[samplePredLiver < 0.50] = 0
        maskLiver[zIndex:maxZ, :, :] = samplePredLiver
        # Spleen
        samplePredSpleen = modelSpleen.predict(sample)
        samplePredSpleen = samplePredSpleen[:,:,:,0]
        samplePredSpleen[samplePredSpleen >= 0.50] = 1
        samplePredSpleen[samplePredSpleen < 0.50] = 0
        maskSpleen[zIndex:maxZ, :, :] = samplePredSpleen

    pixDim = niiInput.header.structarr["pixdim"][1:4]

    maskLiver = np.moveaxis(maskLiver, 0, -1)
    maskLiver = utils.getLargestCC(maskLiver)
    maskLiver = maskLiver * 1
    maskLiver = utils.resizeStack(maskLiver, originalShape, method=cv2.INTER_NEAREST)
    maskLiver = maskLiver.astype(np.uint8) * 255
    liverMetricVolume, liverPixelVolume, liverMedianHU, liverMeanHU, liverDevHU = utils.getMetricsForStack(stackOriginal, maskLiver, pixDim, errodeStruct=np.ones((5,5,1)))
    maskLiver = utils.revertStack(maskLiver, niiInput.header)
    
    niiMaskLiver = nib.Nifti1Image(maskLiver, niiInput.affine)

    maskSpleen = np.moveaxis(maskSpleen, 0, -1)
    maskSpleen = utils.getLargestCC(maskSpleen)
    maskSpleen = maskSpleen*1
    maskSpleen = utils.resizeStack(maskSpleen, originalShape, method=cv2.INTER_NEAREST)
    maskSpleen = maskSpleen.astype(np.uint8) * 255
    spleenMetricVolume, spleenPixelVolume, spleenMedianHU, spleenMeanHU, spleenDevHU = utils.getMetricsForStack(stackOriginal, maskSpleen, pixDim, errodeStruct=np.ones((5,5,1)))
    maskSpleen = utils.revertStack(maskSpleen, niiInput.header)

    niiMaskSpleen = nib.Nifti1Image(maskSpleen, niiInput.affine)

    print("=========================================")
    print("Liver")
    print("=========================================")
    print("Median HU: %0.2f" % liverMedianHU)
    print("Liver Volume (ml): %0.2f" % (liverMetricVolume/1000))
    print("=========================================")
    print("Spleen")
    print("=========================================")
    print("Median HU: %0.2f" % spleenMedianHU)
    print("Liver Volume (ml): %0.2f" % (spleenMetricVolume/1000))

    niiMaskLiver.to_filename(outputLiverMaskFilePath)
    niiMaskSpleen.to_filename(outputSpleenMaskFilePath)
