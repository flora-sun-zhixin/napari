import re
import os
import sys
import copy
import shutil
import json
from glob import glob
from pathlib import Path
from datetime import datetime
import pydicom
import matplotlib.pyplot as plt
loadpath = '/Users/laurayu/Desktop/CV/napari/napari/components/CodeFromFlora'
# loadpath = str(Path().resolve().absolute())
print(loadpath)
with open(os.path.join(loadpath, "config.json")) as File:
    config = json.load(File)

# set up the device =============================== set to 7
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
# # show whether use GPU
# physical_devices = tf.config.list_physical_devices('GPU')
# print("You are using ", physical_devices)
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

import numpy as np
import scipy.io as sio
import h5py
import scipy.io as sio
from tqdm import tqdm
import albumentations as A
import cv2

from napari.components.CodeFromFlora.utils.utils import *
from napari.components.CodeFromFlora.utils.dataPreprocess import *
from napari.components.CodeFromFlora.utils.Losses import Filtered_DC_and_CE_loss
from napari.components.CodeFromFlora.utils.Metrics import MetricsList
from napari.components.CodeFromFlora.models.ResUNet_2D_attention_gate import ResUNet
from napari.components.CodeFromFlora.dataPreprocess.positionalEncoding import PositionEmbeddingSine1DwithSliceNum
from napari.components.CodeFromFlora.browse3 import browse_folder
# random seed
np.random.seed(1234)
tf.random.set_seed(1234)

def predictUnknown(ctData):
    def readCTDataFromDicom(fileList):
        """
        :Params fileList: is a list of all ct dicom files for one mice
        :Returns matrix: 3D CT data, (512, 512, 679)
        :Returns info: a dict that contains information that can be used to resample PET
        """
        # preparation
        matrix = np.zeros((512, 512, len(fileList)))
        intercept = np.zeros((len(fileList),))
        slope = np.zeros((len(fileList),))
        info = {
            "pixel_size_x": set(),
            "pixel_size_y": set(),
            "pixel_size_z": set(),
            "name": set(),
            "origin": None
        }

        printInfo = False
        for fileName in tqdm(fileList, desc="Reading CT data: "):
            ds = pydicom.dcmread(fileName)
            if ds.Modality == "CT":
                if printInfo:
                    print("\nname: ", ds.PatientName)
                    print("energy: ", ds.KVP)
                    print("study date: ", ds.StudyDate)
                    print("acquisition date: ", ds.AcquisitionDate)
                    printInfo = False
                sliceNum = int(ds.InstanceNumber) - 1
                matrix[:, :, sliceNum] = ds.pixel_array.T
                intercept[sliceNum] = ds.RescaleIntercept
                slope[sliceNum] = ds.RescaleSlope
                info["pixel_size_x"].add(ds.PixelSpacing[0])
                info["pixel_size_y"].add(ds.PixelSpacing[1])
                info["pixel_size_z"].add(ds.SliceThickness)
                info["name"].add(ds.StudyDescription)
                if sliceNum == 0:
                    info["origin"] = [ds.ImagePositionPatient]

        matrix = matrix * np.array(slope) + np.array(intercept)
        info["pixel_size"] = np.array([info["pixel_size_x"].pop(),
                                       info["pixel_size_y"].pop(),
                                       info["pixel_size_z"].pop()]).astype(np.float32)  # unit: mm
        info["name"] = info["name"].pop()
        info["origin"] = info["origin"][0]
        del info["pixel_size_x"], info["pixel_size_y"], info["pixel_size_z"]

        return matrix, info

    # specify data path and root path
    # ctFilePath = browse_folder()
    # testList = generateFileListInFolder(ctFilePath)
    # ctData, ctInfo = readCTDataFromDicom(testList)

    weightLoadsPath = os.path.join(loadpath, "logs/best")

    def findBoundAllAxis(ctData):
        """
        ctData: 3D matrix, (axis0, axis1, axis2)
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

        axises = [[0, 1, 2], [1, 0, 2], [2, 1, 0]]
        res = []
        for i in range(3):
            ctDataT = np.transpose(ctData, axes=axises[i])
            res += findBound(np.max(ctDataT, axis=(1, 2)))
        return res

    def cutBound(ctData):
        bound = findBoundAllAxis(ctData)
        ctData = ctData[bound[0]: bound[1], bound[2]:bound[3], bound[4]: bound[5]]
        return ctData, bound

    def cutBound_withBound(ctData, bound):
        ctData = ctData[bound[0]: bound[1], bound[2]:bound[3], bound[4]: bound[5]]
        return ctData

    ctData, bound = cutBound(ctData)
    ctData = np.expand_dims(ctData, axis=-1)

    posEncodingGenerator = PositionEmbeddingSine1DwithSliceNum(num_pos_feats=config["dataset"]["positionalEncoding"],
                                                               normalize=True)
    def addPostionalEncoding(dataInstanceDic):
        posEncoding = posEncodingGenerator.generatePosEncoding(dataInstanceDic)
        dataInstanceDic["positionalEncoding"] = posEncoding
        return dataInstanceDic


    data_kargs = {'ic': config["dataset"]["ic"],
                  'oc': config["dataset"]["oc"],
                  'padding': config["dataset"]["padding"]}

    net_kargs = {'depth': config["model"]["depth"],
                 'start_exp': config["model"]["start_exp"],
                 'num_conv': 2
                 # 'num_head': config["model"]["num_head"]
                 }

    net = ResUNet(**data_kargs, **net_kargs)
    net.loadWeight(weightLoadsPath)

    def removePad(originalLengh, currentLength):
        start = abs(originalLengh - currentLength) // 2
        end = start + originalLengh
        return int(start), int(end)
    def cleanFemurs(predData):
        """
        find where the middle location of spine,
        slidesCode < middle, then 2nd = and(2nd, 3rd channel), 3rd = 0
        slidesCode > middel, then 3rd = and(2nd, 3rd channel), 2nd = 0
        """
        predData = predData.numpy()
        # predData = predData.reshape((512,512,496,7))
        print(np.where(np.sum(predData[:, :, :, 1], axis=(-1, -2)) > 0)[0])

        middle = int(np.mean(np.where(np.sum(predData[:, :, :, 1], axis=(-1, -2)) > 0)))
        # middle = int(np.mean(np.where(np.sum(predData[:, :, :, 1], axis=(-1, -2)) > 0)[0]))
        print(middle)
        for slc in range(len(predData)):
            if slc < middle:
                predData[slc, ..., 2] = np.logical_or(predData[slc, ..., 2], predData[slc, ..., 3])
                predData[slc, ..., 3] = 0
            if slc > middle:
                predData[slc, ..., 3] = np.logical_or(predData[slc, ..., 2], predData[slc, ..., 3])
                predData[slc, ..., 2] = 0

        return predData

    previousSliceNum = -1
    predictList = []
    predictFileList = []
    # testList = list(testList)
    for i in tqdm(range(len(ctData))):
        di = {"image": ctData[i, :, :],
                  "mask": None,
                  "sliceNum": i,
                  "totalSliceNum": ctData.shape[0],
                  "imageShape": ctData.shape[1:],
                  "maskShape": (ctData.shape[1], ctData.shape[2], 7)}

        di = addPostionalEncoding(di)
        # pad to same size
        dividable = 2 ** (config["model"]["depth"] - 1)
        print("find the max shape")
        train_axis0_max = tf.cast(ceilingToProduct(di["image"].shape[0], dividable), tf.float32)
        train_axis1_max = tf.cast(ceilingToProduct(di["image"].shape[1], dividable), tf.float32)
        print("axis0", train_axis0_max, "axis1", train_axis1_max)
        # di = padding(di, train_axis0_max, train_axis1_max)


    # 2D augmentation
        transform = A.Compose([
            A.Rotate(5, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.Flip(0, p=0.5),
            A.Flip(1, p=0.5)
        ])

        def get_augmentation(image, mask, positionalEncoding):
            #    data = {"image": image, "mask": mask}
            image = np.concatenate([image, positionalEncoding], axis=-1)
            aug_data = transform(image=image, mask=mask)
            image, mask = aug_data["image"], aug_data["mask"]
            image = image[..., :1]
            positionalEncoding = image[..., 1:]
            return image, mask, positionalEncoding


        def augmentationWrapper(dataInstance, aug=tf.constant(True)):
            image, mask, positionalEncoding = dataInstance["image"], dataInstance["mask"], dataInstance["positionalEncoding"]
            if aug:
                image, mask, positionalEncoding = tf.numpy_function(func=get_augmentation,
                                                                    inp=[image, mask, positionalEncoding],
                                                                    Tout=[tf.float32, tf.float32, tf.float32])

            image = tf.cast(image, tf.float32)
            dataInstance["image"] = normalize(image)
            return dataInstance


        di = augmentationWrapper(di, aug=tf.constant(False))
        di = padding(di, train_axis0_max, train_axis1_max)
        ho, wo, co = di["maskShape"]
        h, w, c = di["image"].shape
        pred = net.pred(weightLoadsPath, di["image"][tf.newaxis, ...], di["positionalEncoding"][tf.newaxis])
        # remove pad
        axis0_s, axis0_e = removePad(ho, h)
        axis1_s, axis1_e = removePad(wo, w)
        pred = pred[0, axis0_s:axis0_e, axis1_s:axis1_e, :]
        if di["sliceNum"] - previousSliceNum != 1:
            print("one mouse done")
            mouse = testList.pop(0)
            mouse = mouse.split("/")[-1].split(".")[0]
            predict3D = tf.stack(predictList, axis=0)
            predict3D = tf.where(predict3D == tf.reduce_max(predict3D, axis=-1, keepdims=True), 1., 0.)
            predict3D = cleanFemurs(predict3D)
            print(mouse)
            predictFile = os.path.join(loadpath, "results", mouse + "_predData.hdf5")
            predictFileList.append(predictFile)
            with h5py.File(predictFile, "w") as f:
                f.create_dataset("predData", data=predict3D)

            predictList = []
        previousSliceNum = di["sliceNum"]
        predictList.append(pred)

    # mouse = testList.pop(0)
    # mouse = mouse.split("/")[-1].split(".")[0]
    predict3D = tf.stack(predictList, axis=0)
    predict3D = tf.where(predict3D == tf.reduce_max(predict3D, axis=-1, keepdims=True), 1., 0.)
    predict3D = cleanFemurs(predict3D)
    # indices = [i for i in range(tf.shape(predict3D)[-1]) if i != 5]
    # predict3D = tf.gather(predict3D, indices, axis=-1)
    return predict3D
# np.save('outfile_0728_loadpath_otherbone.npy', predict3D)
#
# predictFile = os.path.join(loadpath, "results", mouse + "_predData.hdf5")
# predictFileList.append(predictFile)
# with h5py.File(predictFile, "w") as f:
#     f.create_dataset("predData", data=predict3D)
#
#
# def plotCTandMask(ctData, maskData, slc, ctcmap, maskcmap):
#     """
#     plot the ct and ct with mask and mask itself
#     """
#     # better = np.clip(ctData[slc, :, :], amin, amax)
#     import matplotlib.pyplot as plt
#     import matplotlib
#     mask = np.argmax(maskData[slc, :, :, :], axis=-1).astype(np.float32)
#     mask[mask == 0] = np.nan
#     cmap = matplotlib.cm.get_cmap("bwr").copy()
#     cmap.set_bad("k", alpha=0.)
#     better = ctData[slc, ...]
#     fig = plt.figure()  # dpi=1000)
#     plt.imshow(better, cmap=ctcmap, interpolation="none")
#     plt.imshow(mask, cmap=cmap, alpha=1, vmin=0, vmax=7, interpolation="none")
#     plt.axis("off")
#     plt.title(f"CT with mask - slice{slc}")
#
#     return fig
#
#
# def generateVideo(tmpImagePath, videoSavePath, videoName):
#     import imageio
#     writer = imageio.get_writer(os.path.join(videoSavePath, videoName + ".mp4"), fps=10)
#     fileList = generateFileListInFolder(tmpImagePath)
#     fileList = sorted(fileList, key=lambda x: int(x.split("/")[-1].split("-")[-1].split(".")[0]))
#     for im in fileList:
#         writer.append_data(imageio.imread(im))
#     writer.close()
#     shutil.rmtree(tmpImagePath)
#
#
# # ImgTmpFolder = "/project/cigserver1/export/zhixin.sun/Segmentation/ImageTmp"
# # videoSave = "/project/cigserver1/export/zhixin.sun/Segmentation/saveVideo"
# #
# # for file in predictFileList:
# #     name = file.split("/")[-1].split(".")[0]
# #     predData = read_hdf5_file(file)["predData"]
# #     ctData = read_hdf5_file(os.path.join("/project/cigserver1/export/zhixin.sun/Segmentation/Mice3D_80_new",
# #                                          name.replace("predData", "ctData") + ".hdf5"))["ctData"][..., np.newaxis]
# #     bound = findBoundAllAxis(ctData)
# #     ctData = cutBound(ctData, bound)
# #     if not os.path.exists(ImgTmpFolder):
# #         os.mkdir(ImgTmpFolder)
# #
# #     for slc in tqdm(range(len(ctData)), desc="Generating Plots: "):
# #         fig = plotCTandMask(ctData, predData, slc, "gray", "hot")
# #         fig.savefig(os.path.join(ImgTmpFolder, f"slide-{slc}.png"), bbox_inches="tight")
# #         plt.close()
# #     generateVideo(ImgTmpFolder, videoSave, name)
