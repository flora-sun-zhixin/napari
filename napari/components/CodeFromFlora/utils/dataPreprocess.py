import numpy as np
import os
import re
import sys
import shutil
import h5py
import random
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from glob import glob
# from dataLoads.dataInstance import *

# solving imbalance
def statisticPixel(miceList):
    """
    count the number of pixel for each classes in the training mask, returns dictionary
    """
    count = {"background": 0, "spine": 0, "left_femur": 0, "rt_femur": 0}
    code =  {'left_femur': 3, 'rt_femur': 2, 'spine': 1, "background": 0}
    count_items = ["background", "spine", "left_femur", "rt_femur"]
    for mousePath in miceList:
        maskPath = mousePath[1]
        with h5py.File(maskPath, "r") as f:
            mask = np.array(f["maskData"])
            mask[mask > 3] = 0
        mask = mask.astype(np.int32)
        for item in count_items:
            count[item] += int(np.sum(mask == code[item]))
    return count

def statisticPixel_general(trainDataset, numLabel):
    """
    count the number of pixel for each class in the training mask, returns dictionary
    0 - background, 1,2,... will be count for each labels
    Args:
        trainList: a list of files that contains the mask information
        readinDataFunction: the function that helps to deal with the functions,
                            the output should be a (n+1) channel mask with each channel represent each labels, 
                            0th channel being the background
        num_label: the number of labele including background
    Returns:
        a dictionary where keys are enumerated integers and items are counts for the labels
    """
    count = {}
    # init count
    for i in range(numLabel):
        count[i] = 0
    # read in file one by one and count the num of each label
    print("begin to count num of pixels: ")
    for data in trainDataset:
        mask = data["mask"]
        for i in range(numLabel):
            count[i] += int(tf.reduce_sum(tf.cast(mask[..., i], tf.int64)))

    return count

def statisticSlice_general(trainDataset, numLabel, numSlices):
    """
    count the number of  each slice contains the ROI in training mask, returns array
    0 - background, 1,2,... will be count for each labels
    Args:
        trainDataset
        num_label: the number of labele including background
    Returns:
        a numSlices*numLabel array, with each element be the count for the corresponding item
    """

    count = np.zeros((numSlices, numLabel))
    # read in file one by one and count the num of each label
    print("begin to count for slices ")
    for dataInstance in tqdm(trainDataset):
        image, mask, sliceNum = dataInstance["image"], dataInstance["mask"], dataInstance["sliceNum"]
        if tf.reduce_sum(mask[..., 1:]) == 0:
            count[int(sliceNum), 0] += 1
        else:
            for i in range(1, numLabel):
                if np.any(mask[..., i] == 1):
                    count[int(sliceNum), i] += 1

    return count

def computeLikelihood(count, smooth=1):
    """
    count is a a numSlices * numLabel array
    """
    numSlices = count.shape[0]
    return (count + smooth)/ (np.sum(count, axis=0, keepdims=True) + smooth * numSlices)

def computeWeight(countDic, reverse=True):
    """
    according to the count of pixels, compute the weights that could balance the imbalance.
    """
    count_items = ["background", "spine", "rt_femur", "left_femur"] # order matters, consists with the code
    count = np.array([countDic[key] for key in count_items if key in countDic.keys()])
    if reverse:
        return np.sum(count)/count
    return count / np.sum(count)

def computePairedWeightForROICls(trainDataset, numLabel):
    """
    Count the number for slices that contains each type of ROI
    Args:
        trainDataset
        numLabel: the number of labels including background
    Returns:
        the average ratio between neg and pos for each class
    """
    countPos = tf.zeros((numLabel,), dtype=tf.float32)
    countNeg = tf.zeros((numLabel,), dtype=tf.float32)
    for dataInstance in tqdm(trainDataset):
        ROIClass = tf.cast(dataInstance["class"], tf.float32)
        countPos += ROIClass
        countNeg += 1. - ROIClass

    posWeight = countNeg / countPos

    return posWeight

def computePairedWeightForClassSeg(trainDataset, numLabel):
    """
    compute the average ratio between positive and negative for a slice contains the ROI.
    """
    count = {}
    for i in range(numLabel):
        count[i] = tf.zeros((2,))
    for dataInstance in tqdm(trainDataset):
        mask = tf.cast(dataInstance["mask"], tf.float32)
        for i in range(mask.shape[-1]):
            pos = tf.reduce_sum(mask[..., i])
            if pos > 0:
                neg = tf.reduce_sum(1 - mask[..., i])
                count[i] += tf.convert_to_tensor([pos, neg])
    posWeight = tf.convert_to_tensor([count[i][1] / count[i][0] for i in range(numLabel)])
    return posWeight

def computeWeight_general(countDic, reverse=True):
    """
    according to the count of pixels, compute the weights that could balance the imbalance.
    """
    count = tf.convert_to_tensor([countDic[i] for i in sorted(countDic.keys(), key=lambda x: int(x))])
    if reverse:
        return tf.reduce_sum(count) / count
    return count / tf.reduce_sum(count)

def generateData(miceList, rangeList, dataPath, toPath, rangeName):
    # input @miceList is a list that contains all the mice
    # input @miceRangelist is a ndarray that specifies 
    #                      where for each file we'd like to trunckate.
    # input @dataPath is the path from where we read the data
    # input @toPath is the path where we are gonna write the data
    # input @rangeName is a string that explains where the trunk is for. eg. "spine"
    # hidden output: write the data to "content" dir to speed up the file reading
    # output: a list of file path of the newly writen files in the "content" dir
    #         col0 - ctData
    #         col1 - maskData
    dataNameList = ["ctData", "maskData"]
    dataRefer = ["ctData", "mask"]
    pathList = []
    for i in tqdm(range(len(miceList))):
        mouse = miceList[i]
        rng = rangeList[i, :]
        for j in range(2):
            kind = dataNameList[j]
            rfr = dataRefer[j]
            # read data
            with h5py.File(os.path.join(dataPath, f"{mouse}_{kind}.hdf5"), "r") as f:
                data = np.array(f[rfr])
            if j == 1:
                data[data > 3] = 0
            dataCube = data[rng[0]: rng[1], rng[2]: rng[3], rng[4]: rng[5]]
            # write data
            with h5py.File(os.path.join(toPath, f"{mouse}_{rangeName}_{kind}.hdf5"), "w") as f:
                f.create_dataset(rfr, data=dataCube, dtype=np.float32)
            pathList.append(os.path.join(toPath, f"{mouse}_{rangeName}_{kind}.hdf5"))

    return np.array(pathList).reshape(len(miceList), 2)

def generateFileList(miceList, dataPath):
    dataNameList = ["ctData", "maskData"]
    pathList = []
    for mouse in miceList:
        for i in range(2):
            pathList.append(glob(os.path.join(dataPath, f"{mouse}*{dataNameList[i]}.hdf5"))[0])
    return np.array(pathList).reshape(len(miceList), 2)

def classifyMice(mouseName):
    pattern = re.compile("mpet[0-9]+([ab]).*")
    mouseName = mouseName.split("/")[-1]
    cls = re.match(pattern, mouseName).group(1)
    if cls == "a":
        return 0
    elif cls == "b":
        return 1
    else:
        raise Exception("classifyMice: there is classes that is other than a and b.")

def separateMiceList(miceList):
    aList = []
    bList = []
    for mouseName in miceList:
        code = classifyMice(mouseName)
        if code == 0:
            aList.append(mouseName)
        else:
            bList.append(mouseName)

    return aList, bList
    
def resampleMiceFile(miceList):
    aList, bList = separateMiceList(miceList)
    diff = len(aList) - len(bList)
    bList = bList + list(np.random.choice(np.array(bList), diff, replace=True))
    tmp = aList + bList
    random.shuffle(tmp)
    return tmp

def addHeadOffList(tfRecordsList, seed=None):
    tfRecordsList = list(tfRecordsList) + [pth.replace("MiceSeparate", "MiceSeparate_headOff") for pth in tfRecordsList]
    if seed is not None:
        random.seed(seed)
    random.shuffle(tfRecordsList)
    return tfRecordsList

# pad image
def computeMaxSize2D(dataset):
    axis0List = []
    axis1List = []
    for i in dataset:
        axis0, axis1, axis2 = i["imageShape"]
        axis0List.append(axis0)
        axis1List.append(axis1)
        if axis2 != 1:
            print("axis2 is not right, ", axis2)
    axis0List = np.array(axis0List)
    axis1List = np.array(axis1List)
    return np.max(axis0List), np.max(axis1List)

def ceilingToProduct(axisSize, numToBeDivided):
    num = axisSize // numToBeDivided
    if axisSize % numToBeDivided > 0:
        num += 1
    return num * 16

def generatePadList(imageShape, axis0_max, axis1_max):
    """
    imageShape: (axis0, axis1, channel)
    """
    axis0, axis1 = imageShape[:2]
    axis0_diff = axis0_max - axis0
    axis0_diff_half = axis0_diff // 2
    axis1_diff = axis1_max - axis1
    axis1_diff_half = axis1_diff // 2
    return [[axis0_diff_half, axis0_diff - axis0_diff_half],
            [axis1_diff_half, axis1_diff - axis1_diff_half],
            [tf.cast(tf.constant(0), tf.float32), tf.cast(tf.constant(0), tf.float32)]]

def padding(dataInstance, axis0_max, axis1_max):
    image, mask = dataInstance["image"], dataInstance["mask"]
    padding = generatePadList(dataInstance["imageShape"], axis0_max, axis1_max)
    while tf.reduce_any(tf.concat(padding, 0) < 0):
        axis0_max = tf.cast(axis0_max + 16, tf.float32)
        axis1_max = tf.cast(axis1_max + 16, tf.float32)
        padding = generatePadList(dataInstance["imageShape"], axis0_max, axis1_max)
    dataInstance["image"] = tf.pad(image, padding)
    # dataInstance["mask"] = tf.pad(mask, padding)
    dataInstance["positionalEncoding"] = tf.pad(dataInstance["positionalEncoding"], padding) 
    return dataInstance

# deal with overlap
def abstractOverlap(maskData, kindEnc1, kindEnc2, keepdims=True):
    """
    abstract the overlap part of kindEnc1 and kindEnc2
    :params maskData: 2D image with channel (H, W, C), one hot encoded
    :params kindEnc1: int, between which kind and kindEnc2 we r finding overlap
    :params kindEnc2: int, between which kind and kindEnc1 we r finding overlap
    """
    maskData = tf.cast(maskData, tf.float32)
    overlap = tf.where(tf.logical_and(maskData[..., kindEnc1] > 0.5, maskData[..., kindEnc2] > 0.5),
                       tf.math.maximum(maskData[..., kindEnc1], maskData[..., kindEnc2]),
                       0.)
    if keepdims:
        overlap = overlap[..., tf.newaxis]
    return tf.cast(overlap, maskData.dtype)

def removeOverlap(maskData, kindEnc1, kindEnc2, removeBoth=True):
    """
    remove the overlap bewteen kindEnc1 and kindEnc2, if removeBoth, then the overlap part will be removed from both, if not removeBoth, the overlap will be removed from kindEnc2.
    spine = 1
    rightFemur = 2
    leftFemur = 3
    pelvis = 4
    """
    maskData = tf.cast(maskData, tf.float32)
    # remove the overlap btw femurs and pelvis
    overlap = abstractOverlap(maskData, kindEnc1, kindEnc2, keepdims=True)
    if removeBoth:
        op = tf.cast(
            tf.map_fn(
                lambda i: tf.cast(tf.logical_or(i==kindEnc1, i==kindEnc2), maskData.dtype),
                tf.range(tf.shape(maskData)[-1], dtype=maskData.dtype)),
            dtype=maskData.dtype)
    else:
        op = tf.cast(
            tf.map_fn(
                lambda i: tf.cast(i==kindEnc2, maskData.dtype),
                tf.range(tf.shape(maskData)[-1]), dtype=maskData.dtype),
            dtype=maskData.dtype)

    overlapWhole = overlap * op
    maskData = tf.math.maximum(maskData - overlapWhole, 0.)

    return maskData, overlap
        
def separateOverlap(maskData, kindEnc1, kindEnc2):
    """
    separate the overlap between spine and pelvis
    background 0, spine 1, rightFemur 2, leftFemur 3, pelvis 4
    """
    # separate the overlap btw spine and pelvis
    maskData = tf.cast(maskData, tf.float32)
    maskData, overlap = removeOverlap(maskData, kindEnc1, kindEnc2, removeBoth=True)
    maskData = tf.concat([maskData, overlap], axis=-1)
    return maskData

def putBackOverlap(maskData, overlapEnc, kindEnc1, kindEnc2):
    """
    put the overlap between kindEnc1 and kindEnc2 back to kindEnc1 and kindEnc2
    """
    maskData = tf.cast(maskData, tf.float32)
    overlap = maskData[..., overlapEnc]
    op = tf.cast(
            tf.map_fn(
                lambda i: tf.cast(tf.logical_or(i==kindEnc1, i==kindEnc2), maskData.dtype),
                tf.range(tf.shape(maskData)[-1]), dtype=maskData.dtype),
            dtype=maskData.dtype)
    
    op = overlap[..., tf.newaxis] * op
    maskData = tf.where(tf.logical_or(op > 0.5, maskData > 0.5),
                        tf.math.maximum(op, maskData),
                        maskData)
    # remove the overlap channel from mask
    indices = [i for i in range(tf.shape(maskData)[-1]) if i != overlapEnc]
    maskData = tf.gather(maskData, indices, axis=-1)    
    return maskData
