import re
import os
import shutil
from datetime import datetime
#import h5py
import pydicom
import json
import numpy as np
# import torch
# import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage
import pandas as pd

# # from DicomRTTool.ReaderWriter import DicomReaderWriter
# from modules.utils.visualization import *
# from modules.utils.dataPreprocess import resamplePET
# from modules.readInData import *
# from modules.utils.fileManipulation import generateFileListInFolder
from pathlib import Path
def align(ctData,petData):
    def draw_target_source(fixedImage, ImageBeforeMove, ImageAfterMove,
                           alpha_fixedImage=1, alpha_movedImage=0.5, cmap_fixedImage="gray", cmap_movedImage="hot", petMin=None):
        """
        plot the median slice of pet/ct from 3 perspective before registration and after registration
        """
        def findMiddle(image, axis):
            return np.take(image, image.shape[axis] // 2, axis=axis)

        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        if petMin is not None:
            ImageBeforeMove = np.ma.masked_where(ImageBeforeMove == petMin, ImageBeforeMove)
            ImageAfterMove = np.ma.masked_where(ImageAfterMove == petMin, ImageAfterMove)

        for i in range(3):
            axes[0, i].imshow(findMiddle(fixedImage, i), cmap=cmap_fixedImage, alpha=alpha_fixedImage)
            axes[0, i].imshow(findMiddle(ImageBeforeMove, i), cmap=cmap_movedImage, alpha=alpha_movedImage)
            axes[0, i].axis("off")
            axes[1, i].imshow(findMiddle(fixedImage, i), cmap=cmap_fixedImage, alpha=alpha_fixedImage)
            axes[1, i].imshow(findMiddle(ImageAfterMove, i), cmap=cmap_movedImage, alpha=alpha_movedImage)
            axes[1, i].axis("off")
        for ax, row in zip(axes[:, 0], ["Before", "After"]):
            ax.set_ylabel(row, rotation=0, size='large')

        fig.tight_layout()
        return fig
    def plotCTandPET(ctData, petData, slc, ctcmap, petcmap, petMin=0.02, petMax=0.04):
        cutCT = np.clip(ctData[slc, :, :], -400, 600)
        cutPET = np.clip(petData[slc, :, :], petMin, petMax)
        masked = np.ma.masked_where(petData[slc, :, :] < 0, petData[slc, :, :])
        fig = plt.figure()
        plt.imshow(cutCT, cmap=ctcmap, interpolation="none")
        plt.imshow(masked, cmap=petcmap, interpolation="none", alpha=0.7, vmax=petMax, vmin=petMin)
        plt.axis("off")
        plt.title(f"CT with PET - slice{slc}")
        return fig
    # # ==========================================================================================
    # # this file servers for manually register pet and ct and save them into a json file
    # ctPath = "/Users/flora/Docs/CIG/Segmentation/NewCT"
    # petPath = "/Users/flora/Docs/CIG/Segmentation/NewPET"
    recordJsonFile = "./register.json"
    moveRecord = {}
    if os.path.exists(recordJsonFile):
        with open(recordJsonFile, "r") as f:
            moveRecord = json.load(f)

    # for ctFile in sorted(generateFileListInFolder(ctPath)):
    #     if ctFile.split("/")[-1] in moveRecord.keys():
    #         continue
    #     petFile = getPETFileName(ctFile, petPath)
#    ctData = np.load('./M1/ctData.npy')
#    petData = np.load('./M1/petData.npy')
    print("Begin align")# for {ctFile.split('/')[-1]} and {petFile.split('/')[-1]}")
        # data = read_hdf5_file(ctFile)
        # ctData, ctInfo = data["ctData"], getInfo(data)
        # data = read_hdf5_file(petFile)
        # petData, petInfo = data["petData"], getInfo(data)
        # petData = resamplePET(petData, (petInfo["pixel_size"][::-1]).tolist(),
#        #                       targetSize=ctData.shape[::-1], targetSpace=ctInfo["pixel_size"][::-1])
    petMin, petMax = np.quantile(petData, (0,1))
    petData = np.clip(petData, petMin, petMax)
    aligned = False
    while not aligned:
        move = input("input the num of pixel you would like to move the mask: \n")
        move = [int(piece.strip()) for piece in move.split(",")]
        petDataMoved = np.roll(petData, move[0], axis=0)
        petDataMoved = np.roll(petDataMoved, move[1], axis=1)
        petDataMoved = np.roll(petDataMoved, move[2], axis=2)
        draw_target_source(ctData, petData, petDataMoved, alpha_movedImage=0.5, petMin=petMin)
        plt.show(block=False)
        plt.waitforbuttonpress()
        #plt.pause(1)
        plt.close()
        aligned = True if (input("Is the align good enough: \n").strip() == "y") else False
        if aligned:
            for i in range(150, 350, 3):
                plotCTandPET(ctData, petDataMoved, i, "gray", "hot", petMin, petMax)
                plt.show(block=False)
                plt.waitforbuttonpress()
                #plt.pause(10)
                plt.close()
            aligned =True if (input("Do you want further adjust it?\n").strip() == "n") else False

    moveRecord['0'] = move
    return move
    with open(recordJsonFile, "w") as f:
        f.write(json.dumps(moveRecord))

