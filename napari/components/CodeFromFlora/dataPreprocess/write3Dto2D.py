import os
import shutil
import re
from glob import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import scipy.io as sio
import h5py
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
from tqdm import tqdm

os.sys.path.append("/project/cigserver1/export/zhixin.sun/Segmentation/SegCode")
from utils.utils import generateFileListInFolder

# parse data to byte
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
    """make all the feature into a long series of bytes"""
    array = tf.io.serialize_tensor(array)
    return array

# write data to Features and Examples
def parse_single_image(image, mask, sliceNum, totalSliceNum, mouseName):
    """define the feature writing into the byte format"""
    #define the dictionary -- the structure -- of our single example
    data = {
        'height' : _int64_feature(image.shape[0]),
        'width' : _int64_feature(image.shape[1]),
        'image_depth' : _int64_feature(image.shape[2]),
        'mask_depth' : _int64_feature(mask.shape[2]),
        'raw_image' : _bytes_feature(serialize_array(image)), # float32
        'raw_mask' : _bytes_feature(serialize_array(mask)),    # int8
        'sliceNum' : _int64_feature(sliceNum),
        'totalSliceNum':  _int64_feature(totalSliceNum),
        'mouseName': _bytes_feature(mouseName.encode())
    }
    #create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))
    
    return out

# write 2D data to tfrecords, per mouse one file
def write_images_to_tfr_per_mouse(images, masks, filename:str="images"):
    """write every single slices of images and masks into tfrecords files"""
    filename= filename + ".tfrecords"
    writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
    count = 0
    
    for index in range(len(images)):
        #get the data we want to write
        current_image = images[index] 
        current_mask = masks[index]
        
        out = parse_single_image(image=current_image, mask=current_mask, sliceNum=index, totalSliceNum=len(images),
                                 mouseName=generate2DfileName(filename))
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count

def readinMiceDataHdf5(dataFile, labeled=True):
    """
    read in data, return x and y, y is one hot encoded.
    """
    with h5py.File(dataFile[0], "r") as f:
        x = np.array(f["ctData"]).astype(np.float32)

    y = None
    if labeled:
        with h5py.File(dataFile[1], "r") as f:
            y = np.array(f["maskData"]).astype(np.int8)
    else:
        y = np.zeros((x.shape[0], x.shape[1], x.shape[2], 6)).astype(np.int8)

    return x[..., np.newaxis], y

def generate2DfileName(fileName3D):
    """
    Given the 3D file name and which slice this file contains, generate the file name for this 2D data mat file
    """
    pattern = re.compile(".*/(mpet.*M[0-9]).*")
    mouseName = pattern.match(fileName3D).group(1)
    return mouseName

def generate3DfileList(dataPath):
    """ generate a list of 3D data that need to be processed."""
    ctFiles = list(glob(os.path.join(dataPath, "*_ctData.hdf5")))
    maskFiles = [fn.replace("ctData", "maskData") for fn in ctFiles]
    dataList = np.c_[np.array(ctFiles).reshape(-1, 1), np.array(maskFiles).reshape(-1, 1)]
    return dataList

def writeMiceDatasetTo2D(dataFileList, toDataPath, readinDataFunction, xndim, yndim, sliceAxis, headOff=True):
    """
    Given a list of 3D data files, write sliced 2D files into the toDataPath.
    Args:
        dataFileList: a list of 3D file that need to be writen into 2D files
        toDataPath: the data path into which we write the 2D data file
        readinDataFunction: how we read in the 3D data file
        xndim: the ndim of image
        yndim: the ndim of mask
        sliceAxis: along which axis we do the slicing
    Returns:
       None 
    """
    # generate transpose orders
    xtrans = [sliceAxis] + [i for i in range(xndim) if i != sliceAxis]
    ytrans = [sliceAxis] + [i for i in range(yndim) if i != sliceAxis]

    if headOff:
        start = (679 - 496) // 2
        end = start + 496
     
    for dataFile in tqdm(dataFileList):
        dataX, dataY = readinDataFunction(dataFile, labeled=True)

        if headOff:
            dataX = dataX[:, :, start: end, :]
            dataY = dataY[:, :, start: end, :]
        bound = findBoundAllAxis(dataX)
        dataX = cutBound(dataX, bound)
        dataY = cutBound(dataY, bound)
        
        dataX = np.transpose(dataX, xtrans)
        dataY = np.transpose(dataY, ytrans)
        write_images_to_tfr_per_mouse(dataX, dataY, os.path.join(toDataPath, generate2DfileName(dataFile[0])))
    return

def normalize(image):
    minValue = np.min(image)
    maxValue = np.max(image)
    return (image - minValue) / (maxValue - minValue)

def standardize(image):
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / std

def findBoundAllAxis(ctData):
    """
    ctData: 3D matrix, (axis0, axis1, axis2, channel)
    Find the range where actually contains the ct info, remove the bound
    """
    def findBound(maxValueArray):
        """
        maxValueArray: a 1D array that represents the max value along each axis, 
        then find the index where the max value > 0, trying to remove the thick 
        bound that does not contains info here.
        """
        index = np.arange(len(maxValueArray))
        mask = np.where(maxValueArray > 1, 1, 0)
        end = np.max(index * mask)
        strt = np.min(index * mask + 1e5 * (1 - mask)) 
        return sorted([int(strt), int(end)])

    axises = [[0, 1, 2, 3], [1, 0, 2, 3], [2, 1, 0, 3]]
    res = []
    for i in range(3):
        ctDataT = np.transpose(ctData, axes=axises[i])
        res += findBound(np.max(ctDataT, axis=(1, 2, 3)))
    return res

def cutBound(data, bound):
    data = data[bound[0]: bound[1], bound[2]:bound[3], bound[4]: bound[5], ...]
    return data

# main
toDataPath = "/project/cigserver1/export/zhixin.sun/Segmentation/Mice2D_80"
# toDataPath = "/export/project/zhixin.sun/Segmentation/MiceSeparate"
# toDataPath = "/Users/flora/Docs/CIG/Segmentation/MiceSeparate" # local
# remove the old toDataPath and set up a new one
if os.path.exists(toDataPath):
    shutil.rmtree(toDataPath)
os.makedirs(toDataPath)

# write the data into the path
dataPath = "/project/cigserver1/export/zhixin.sun/Segmentation/Mice3D_80_hdf5"
# dataPath = "/Users/flora/Docs/CIG/Segmentation/Mice3D"
dataFileList = generate3DfileList(dataPath)
# filtered = []
# doneGroup1 = re.compile(".+mpet4522.+")
# doneGroup2 = re.compile(".+mpet4529.+")

# for f in dataFileList:
#     if doneGroup1.match(f[0]):
#         continue
#     if doneGroup2.match(f[0]):
#         continue
#     halfDoneGroup = re.compile(".+mpet4619.+")
#     doneMice = re.compile(".+M[123456].+")
#     if halfDoneGroup.match(f[0]) and doneMice.match(f[0]):
#         continue
#     filtered.append(f)
# filtered = np.array(filtered)
writeMiceDatasetTo2D(dataFileList, toDataPath, readinDataFunction=readinMiceDataHdf5, xndim=4, yndim=4, sliceAxis=0, headOff=False)


# back up for bone and muscle
####################################################################################################
##                                      Bone and Muscle                                           ##
####################################################################################################
def write_images_to_tfr_per_boneAndMuscle(images, masks, filename:str="images"):
    filename= filename + ".tfrecords"
    writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
    count = 0
    
    for index in range(len(images)):
        #get the data we want to write
        current_image = images[index] 
        current_mask = masks[index]
        
        out = parse_single_image(image=current_image, mask=current_mask)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count

def generate3DfileListBoneAndMuscle(dataPath):
    fileList = []
    for subfolder in generateFileListInFolder(dataPath, containsDir=True):
        for fileName in generateFileListInFolder(subfolder, extension="mat"):
            fileList.append(fileName)

    return fileList

def generate2DfileNameBoneAndMuscle(fileName3D):
    import re
    pattern = re.compile(".*/(.*)\sDATASET/(.*)\.mat$")
    dataset, case = pattern.match(fileName3D).group(1, 2)
    return dataset + "-" + case

def readinBoneAndMuscleMat(dataFile):
    """
    read in data, return x and y, y is one hot encoded.
    """
    import scipy.io as sio
    data = sio.loadmat(dataFile)
    caso = data["CASO"][..., np.newaxis]
    bone = data["GT_BONE"][..., np.newaxis]
    muscle = data["GT_MUSCLE"][..., np.newaxis]
    background = (1 - bone) * (1 - muscle)
    return data["CASO"][..., np.newaxis].astype(np.float32), np.concatenate([background, bone, muscle], axis=-1).astype(np.int8)

def writeBoneAndMuscleDatasetTo2D(dataFileList, toDataPath, readinDataFunction, xndim, yndim, sliceAxis):
    # generate transpose orders
    xtrans = [sliceAxis] + [i for i in range(xndim) if i != sliceAxis]
    ytrans = [sliceAxis] + [i for i in range(yndim) if i != sliceAxis]
    for dataFile in tqdm(dataFileList):
        dataX, dataY = readinDataFunction(dataFile)
        dataX = normalize(standardize(dataX))
        dataX = np.transpose(dataX, xtrans)
        dataY = np.transpose(dataY, ytrans)
        write_images_to_tfr_per_boneAndMuscle(dataX, dataY, os.path.join(toDataPath, generate2DfileNameBoneAndMuscle(dataFile[0])))
    return

