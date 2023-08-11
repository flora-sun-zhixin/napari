"""
This file is trying to find the mask range for the 3D mice, so we can crop the head and tail off to save memory
"""
import os
import shutil
import re

import h5py
import numpy as np
import scipy.io as sio
from tqdm import tqdm

os.sys.path.append("/export/project/zhixin.sun/Segmentation/SegCode")
from utils.utils import generateFileListInFolder

def readinMiceDataHdf5(dataFile):
    """
    read in data, return ctData(axis0, axis1, axis2) and y(axis0, axis1, axis2), y is enumerated.
    """
    with h5py.File(dataFile[0], "r") as f:
        ctData = np.array(f["ctData"])
    with h5py.File(dataFile[1], "r") as f:
        mask = np.array(f["maskData"])
    mask[mask > 3] = 0 # remove the spleender and shoulder if exists
    return ctData, mask

def findMaskRange(mask, coordinateMatrix):
    mask[mask > 0] = 1
    mask = mask[..., np.newaxis]
    aux = mask * coordinateMatrix
    axes = tuple(range(coordinateMatrix.ndim - 1))
    maxCoordinate = np.max(aux, axis=axes)
    invMask = 1 - mask
    minCoordinate = np.min(aux + 1e5 * invMask, axis = axes)
    return minCoordinate, maxCoordinate

def makeCoordinateMatrix(shape):
    """ 
    return a matrix in the same shape with each elemens in the cooresponding channel is the corresponding coordinate
    """
    base = np.ones(shape)
    coordinate = []
    for axis in range(len(shape)):
        coordinate.append(np.cumsum(base, axis=axis) - 1)

    return np.stack(coordinate, axis=-1)

# main
dataPath = "/export/project/zhixin.sun/Segmentation/Mice3D_New"
dataFile = [os.path.join(dataPath, "mpet4422b_ct1_M5_ctData.hdf5"),
            os.path.join(dataPath, "mpet4422b_ct1_M5_maskData.hdf5")]
_, mask = readinMiceDataHdf5(dataFile)
coordinateMatrix = makeCoordinateMatrix(mask.shape)
print(mask.shape)
minCoor, maxCoor = findMaskRange(mask, coordinateMatrix)
print(minCoor)
print(maxCoor)

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(mask[minCoor[0], :, :])
plt.title(f"slice{minCoor[0]}")
plt.figure()
plt.imshow(mask[minCoor[0] - 1, :, :])
plt.title(f"slice{minCoor[0] - 1}")
plt.figure()
plt.imshow(mask[maxCoor[0], :, :])
plt.title(f"slice{maxCoor[0]}")
plt.figure()
plt.imshow(mask[maxCoor[0] + 1, :, :])
plt.title(f"slice{maxCoor[0] + 1}")

plt.figure()
plt.imshow(mask[:, minCoor[1], :])
plt.title(f"slice{minCoor[1]}")
plt.figure()
plt.imshow(mask[:, minCoor[1] - 1, :])
plt.title(f"slice{minCoor[1] - 1}")
plt.figure()
plt.imshow(mask[:, maxCoor[1], :])
plt.title(f"slice{maxCoor[1]}")
plt.figure()
plt.imshow(mask[:, maxCoor[1] + 1, :])
plt.title(f"slice{maxCoor[1] + 1}")

plt.figure()
plt.imshow(mask[:, :, minCoor[2]])
plt.title(f"slice{minCoor[2]}")
plt.figure()
plt.imshow(mask[:, :, minCoor[2] - 1])
plt.title(f"slice{minCoor[2] - 1}")
plt.figure()
plt.imshow(mask[:, :, maxCoor[2]])
plt.title(f"slice{maxCoor[2]}")
plt.figure()
plt.imshow(mask[:, :, maxCoor[2] + 1])
plt.title(f"slice{maxCoor[2] + 1}")
