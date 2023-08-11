# CIG Flora Sun 2023
import re
import os
import shutil
from io import StringIO
from datetime import datetime
from datetime import timedelta
from napari.components.CodeFromFlora.predictUnknown3 import predictUnknown
from napari.components.CodeFromFlora.browse3 import browse_folder
from napari.components.CodeFromFlora.manualRegister import align
#import h5py
import numpy as np
# import sys
# np.set_printoptions(threshold=sys.maxsize)
from tqdm import tqdm
import pydicom
from nibabel import ecat
#from modules.utils.fileManipulation import *
import napari
import SimpleITK as sitk

def readHdf5File(fileName, keyName, dtype):
    with h5py.File(fileName, "r") as f:
        data = np.array(f[keyName]).astype(dtype)
    return data

def read_hdf5_file(file_path):
    data = {}
    with h5py.File(file_path, "r") as f:
        keys = list(f.keys())
        for key in keys:
            data[key] = np.array(f[key], dtype=f[key].dtype)

    return data

def parseTrfFile(trfFileName):
    info = {}
    finished = False
    key = ""
    value = ""

    with open(trfFileName, "r") as f:
        for line in f.readlines():
            if "=" in line:
                sp = line.split("=")
                key, value = sp[0], sp[1]
            if "=" not in line and not line.strip().endswith(";"):
                finished = False
                value = value + line
            if line.strip().endswith(";"):
                finished = True
                info[key]= value.strip()
    return info

def readInMatrix(trfFileName):
    info = parseTrfFile(trfFileName)
    tranMatrix = info["TARGET_TRANSFORM_MATRIX"]
    tranMatrix = tranMatrix.replace(",\n", "\n")
    return np.genfromtxt(StringIO(tranMatrix), delimiter=',')

def checkConsistency(infoDict):
    """
    check that the info abstracted is the same for all the pet files, if not throw out an exception
    """
    for key, value in infoDict.items():
        if len(value) == 1:
            continue
        else:
            raise Exception("The meta info reading seems to have different values for {}: expecting 1 value, got {:n}".format(key, len(value)))

def readPETDataFromDicom(fileList):
    """
    :Params fileList: is a list of all pet dicom files for one mice
    :Returns matrix: 3D PET data, (128, 128, 159)
    :Returns info: a dict that contains information that can be used to compute SUV
    """
    matrix = np.zeros((128, 128, len(fileList)))
    intercept = np.zeros((len(fileList),))
    slope = np.zeros((len(fileList),))
    info = {
        "weight": set(),
        "scan_time": set(),
        "injection_time": set(),
        "injection_dose": set(),
        "radio_half_life": set(),
        "pixel_size_x": set(),
        "pixel_size_y": set(),
        "pixel_size_z": set(),
        "name": set()
    }

    # abstract info we need
    for fileName in tqdm(fileList, desc="Reading PET data: "):
        ds = pydicom.dcmread(fileName)                
        if ds.Modality == "PT":
            sliceNum = int(ds.InstanceNumber) - 1
            matrix[:, :, sliceNum] = ds.pixel_array.T
            intercept[sliceNum] = ds.RescaleIntercept
            slope[sliceNum] = ds.RescaleSlope
            info["weight"].add(ds.PatientWeight)
            info["scan_time"].add(ds.AcquisitionDateTime)
            info["injection_time"].add(ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartDateTime)
            info["injection_dose"].add(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
            info["radio_half_life"].add(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
            info["pixel_size_x"].add(ds.PixelSpacing[0])
            info["pixel_size_y"].add(ds.PixelSpacing[1])
            info["pixel_size_z"].add(ds.SliceThickness)
            info["name"].add(ds.StudyDescription)

    # post process
    try:
        checkConsistency(info)
    except Exception as e:
        print(str(e))
        exit()
    matrix = matrix * np.array(slope) + np.array(intercept)
    info["weight"] = info["weight"].pop() * 1000 # unit: g
    info["scan_time"] = datetime.strptime(info["scan_time"].pop(), "%Y%m%d%H%M%S.%f%z")
    info["injection_time"] = datetime.strptime(info["injection_time"].pop(), "%Y%m%d%H%M%S.%f%z")
    info["injection_dose"] = info["injection_dose"].pop() / 37 # unit: nCi
    info["radio_half_life"] = info["radio_half_life"].pop() / 3600 # unit: hours
    info["pixel_size"] = np.array([info["pixel_size_x"].pop(),
                                   info["pixel_size_y"].pop(),
                                   info["pixel_size_z"].pop()]).astype(np.float32) # unit: mm
    info["name"] = info["name"].pop()
    del info["pixel_size_x"], info["pixel_size_y"], info["pixel_size_z"]

    return matrix, info

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
                printInfo=False
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

    # post process
    try:
        checkConsistency(info)
    except Exception as e:
        print(str(e))
        exit()
    matrix = matrix * np.array(slope) + np.array(intercept)
    info["pixel_size"] = np.array([info["pixel_size_x"].pop(),
                                   info["pixel_size_y"].pop(),
                                   info["pixel_size_z"].pop()]).astype(np.float32) # unit: mm
    info["name"] = info["name"].pop()
    info["origin"] = info["origin"][0]
    del info["pixel_size_x"], info["pixel_size_y"], info["pixel_size_z"]
    return matrix, info      

def bqmlToSUV(petDicom, info):
    # from bq/ml to nCi
    petDicom = petDicom / 37
    # compute time delta between injection and scan in hours
    deltaTime = (info["scan_time"] - info["injection_time"]) / timedelta(hours=1)
    # compute the decay factor of the injection
    decayFactor = np.exp(deltaTime * (-0.69314718 / info["radio_half_life"]))
    # compute the injection amount
    injectActivity = info["injection_dose"] * decayFactor
    suv = petDicom / injectActivity * info["weight"]
    return suv
def resamplePET(petData, petSpace, interpolateMethod=sitk.sitkLinear, targetSize=(679, 512, 512), targetSpace=((0.195682, 0.195682, 0.195682))):
    """
    resample the PET to make it has the same space and size with the CT data.
    """
    image = sitk.GetImageFromArray(petData)
    image.SetSpacing(petSpace)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetSize(targetSize)
    resampler.SetOutputSpacing(targetSpace)
    resampler.SetOutputPixelType(sitk.sitkFloat64)
    resampler.SetInterpolator(interpolateMethod)
    newPET = resampler.Execute(image)
    newPET = sitk.GetArrayFromImage(newPET)
    return newPET
def readMaskData(fileName, origin, space):
    mouseNamePattern= re.compile("^Study\sDescription:,(.+).ct")
    roiPattern = re.compile("^ROI\sName:\s(.+)")
    mouseName = ""
    record = False
    data = {}
    with open(fileName, "r") as f:
        for line in f.readlines():
            # find the name of the mouse
            if mouseNamePattern.match(line):
                mouseName = mouseNamePattern.match(line).group(1)
                continue
            # find the kind of the ROI
            if(roiPattern.match(line)):
                print("begin to record data for ", line.strip())
                record = True
                kind = roiPattern.match(line).group(1)
                data[kind] = []
                continue
            # record if not empty line
            if(record):
                if len(line.strip()) != 0:
                    try:
                        data[kind].append([float(d) for d in line.strip().split(",")])
                    except:
                        print("Fail to put ", line, " into pixel data")
    origin = np.array(origin)
    space = np.array(space)
    tem1 = np.r_[origin, 0]
    tem2 = np.r_[space, 1]
    code = {"spine": 1, "rt_femur":2, "lf_femur":3, "pelvis": 4, "otherbone":5, "overlap": 6}
    reverseCode = {value:key for key, value in code.items()}
    # mask = np.zeros((364 - 146, 342 - 170, 679))
    allMask = []
    for i in range(1, 7):
        mask = np.zeros((512, 512, 496))
        pixel = (np.array(data[reverseCode[i]]) - tem1) / tem2
        for pix in pixel:
            mask[round(pix[0]), round(pix[1]), round(pix[2])] = 1
        allMask.append(mask)
    allMask = np.stack(allMask, axis=-1)
    mask = np.ones((512, 512, 496)) - np.where(np.sum(allMask, axis=-1) > 0, 1, 0)
    mask = np.concatenate([mask[..., np.newaxis], allMask], axis=-1)
    code["maskData"] = mask
    return code, mouseName.strip()
def findBoundAllAxis(ctData):
    """
    ctData: 3D matrix, (axis0, axis1, axis2)
    Find the range where actually contains the ct info, remove the bound
    """
    def findBound(maxValueArray):
        """
        maxValueArray: a 1D array that represents the max value along each axis,
        then find the index where the max value > 0, trying to remove the thick
        bound that does not contain info here.
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

### Laura

def plotCT(cponly, M=None):
    if cponly == True:
        #read CT file
        # directory_path = '/Users/laurayu/Desktop/CV/M1/CT/'
        print('please select the folder containing your CT dicom data')
        directory_path = browse_folder()
        f_list = os.listdir(directory_path)
        file_path = [None] * len(f_list)
        n = 0
        for file_name in f_list:
            file_path[n] = os.path.join(directory_path, file_name)
            n += 1
        M, inform = readCTDataFromDicom(file_path)

        # read PET file
        # directory_path_pet = '/Users/laurayu/Desktop/CV/M1/PET/'
        print('please select the folder containing your PET dicom data')
        directory_path_pet = browse_folder()
        fpet_list = os.listdir(directory_path_pet)
        file_path_pet = [None] * len(fpet_list)
        n = 0
        for file_name in fpet_list:
            file_path_pet[n] = os.path.join(directory_path_pet, file_name)
            n += 1
        M_pet, inform_pet = readPETDataFromDicom(file_path_pet)
        suv = bqmlToSUV(M_pet, inform_pet)
        largePET = resamplePET(suv, inform_pet['pixel_size'][::-1].astype(np.float64),
                               interpolateMethod=sitk.sitkLinear, targetSize=M.shape[::-1],
                               targetSpace=(inform["pixel_size"][::-1].astype(np.float64)))
        adj = [10, 2, -55]
        largePET = np.roll(largePET, adj[0], axis=0)
        largePET = np.roll(largePET, adj[1], axis=1)
        largePET = np.roll(largePET, adj[2], axis=2)
        M, bound = cutBound(M)
        largePET = cutBound_withBound(largePET, bound)

        return(M,largePET)
    else:
        mymask = predictUnknown(M)
        # mymask = np.load('/Users/laurayu/Desktop/CIG/CodeFromFlora/outfile_0728_loadpath_otherbone.npy')
        maskcode = mymask

        return(maskcode)
