import re
import os
import shutil
import numpy as np
import h5py
from tqdm import tqdm
from nibabel import ecat
import matplotlib.pyplot as plt

def generateFileListInFolder(folderPath, extension=None, omitHidden=True, absPath=True, containsDir=False):
    """
    generate all the direct file(and subdirectors) under the given folderPath into a list.
    Args:
        folderPath: The abs path of the folder that we want to list the content of
        extension: just list all the file with the extension if extension is not None
        omitHidden: whether omit hidden files
        absPath: whether the items in the list contains the absPath of the file or just a list of filenames
        containsDir: whether the list contains subdirectors or just files
    Returns:
        a list of the content of the folder
    """
    fileList = []
    for f in os.listdir(folderPath):
        if omitHidden and f.startswith("."):
            continue
        if not containsDir and os.path.isdir(os.path.join(folderPath, f)):
            continue
        if extension is not None:
            if f.endswith(extension):
                fileList.append(f)
            else:
                continue
        else:
            fileList.append(f)
    if absPath:
        fileList = [os.path.join(folderPath, f) for f in fileList]

    return fileList

def generateFileNameForEcatFile(fileName):
    print(fileName)
    pattern = re.compile("(mpet[0-9]+[ab]_(ct|em)1)(_v1)?(_M[0-9])\.(ct|pet).img")
    res = pattern.match(fileName).groups()
    return (res[0] + res[3] + "_" + res[4] + "Data.v")

def findKind(fileName):
    print(fileName)
    pattern = re.compile(".+\.(ct|pet)\.img$")
    if pattern.match(fileName).group(1) == "pet":
        return False
    elif pattern.match(fileName).group(1) == "ct":
        return True
    else:
        print(fileName + "goes to neither")

def normalize(image):
    minValue = np.min(image)
    maxValue = np.max(image)
    return (image - minValue) / (maxValue - minValue)

def standardize(image):
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / std

def findBound(maxValueArray):
    index = np.arange(len(maxValueArray))
    mask = np.where(maxValueArray > 0, 1, 0)
    end = np.max(index * mask)
    strt = np.min(index * mask + 1e5 * (1 - mask)) 
    return sorted([int(strt), int(end)])

def findBoundAllAxis(ctData):
    axises = [[0, 1, 2], [1, 0, 2], [2, 1, 0]]
    res = []
    for i in range(3):
        ctDataT = np.transpose(ctData, axes=axises[i])
        res += findBound(np.max(ctDataT, axis=(1, 2)))
    return res

def findROIPath(fileName, maskDataPath):
    pattern = re.compile(".*mpet([0-9]+).*_(M[0-9]).*")
    group = pattern.match(fileName).groups()
    return os.path.join(maskDataPath, group[0], group[1], "other", "ROI_voxel.csv")

def readMaskData(fileName, cropRange):
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

    origin = np.array([-47.727731, -50.354331, -63.525938])
    space = np.array([1.95682E-1, 1.95682E-1, 1.95682E-1])    
    tem1 = np.r_[origin, 0]
    tem2 = np.r_[space, 1]
    code = {"spine": 1, "rightFemur":2, "leftFemur":3}
    # mask = np.zeros((cropRange[1] - cropRange[0], cropRange[3] - cropRange[2], cropRange[5] - cropRange[4]))
    mask = np.zeros((512, 512, 679))
    for key in data.keys():
        pixel = (np.array(data[key]) - tem1) / tem2
        for pix in pixel:
            mask[int(pix[0]) + 1, int(pix[1]), int(pix[2])] = code[key]
    mask = np.flip(mask, axis=0)
    mask = mask[cropRange[0]:cropRange[1], cropRange[2]:cropRange[3], cropRange[4]:cropRange[5]]
    # mask[int(pix[0] - 6 - cropRange[0]), int(pix[1] - cropRange[2]), int(pix[2] - cropRange[4])] = code[key]
    code["maskData"] = mask
    return code, mouseName.strip()

def plotCTandMask(ctData, maskData, slc, ctcmap, maskcmap, CTFileName):
    """
    plot the ct and ct with mask and mask itself
    """
    pattern = re.compile("(mpet.*)_ctData.*")
    mouse = pattern.match(CTFileName).group(1)
    # fig1 = plt.figure() # plt.figure(dpi=1000, figsize=(4, 7))
    amin, amax = np.quantile(ctData[slc, :, :], (0.2, 1))
    better = np.clip(ctData[slc, :, :], amin, amax)
    # plt.imshow(better, cmap=ctcmap, interpolation="none")
    # plt.axis("off")
    # plt.title(f"{mouse} - CT - slice{slc}", fontsize=8)

    fig2 = plt.figure() # plt.figure(dpi=1000, figsize=(4, 7))
    mask = maskData[slc, :, :]
    masked = np.ma.masked_where(mask == 0, mask)
    plt.imshow(better, cmap=ctcmap, interpolation="none")
    # plt.imshow(masked, cmap="Greens", alpha=0.75, vmin=-1, vmax=3, interpolation="none")
    plt.imshow(masked, cmap="Greens", alpha=0.7, interpolation="none")
    plt.axis("off")
    plt.title(f"{mouse} - CT with mask - slice{slc} - new", fontsize=8)

    # fig3 = plt.figure() # plt.figure(dpi=1000, figsize=(4, 7))
    # plt.imshow(mask, cmap=maskcmap, interpolation="none")
    # plt.axis("off")
    # plt.title(f"{mouse} - mask - slice{slc} - new", fontsize=8)

    return fig2 # fig1, fig2, fig3

dataRootPath = "/Users/flora/Docs/CIG/Segmentation/OriginalMiceData/Used"
temDataPath = "/Users/flora/Docs/CIG/Segmentation/OriginalMiceData/ecat"
maskDataPath = "/Users/flora/Docs/CIG/Segmentation/OriginalMiceData/remarkData"
toDataPath = "/Users/flora/Docs/CIG/Segmentation/Mice3D_img"

if os.path.exists(toDataPath):
    shutil.rmtree(toDataPath)
os.mkdir(toDataPath)

# transfer the img to ecat
# folderList = generateFileListInFolder(dataRootPath, containsDir=True)
# for folder in folderList:
#     fileList = generateFileListInFolder(folder, extension="img")
#     for fileName in fileList:
#         subfolder = "CT" if findKind(fileName) else "PET"
#         ecatName = os.path.join(temDataPath, subfolder, generateFileNameForEcatFile(fileName.split("/")[-1]))
#         os.system(f"upet2e7 {fileName} {ecatName}")

fileList = sorted(generateFileListInFolder(temDataPath + "/CT"))
# fig = plt.figure(figsize = (7,7))
for fileName in tqdm(fileList):
    img = ecat.load(fileName)
    frame0 = img.get_frame(0)
    res = findBoundAllAxis(frame0)
    print("ct range: ", res)
    frame0 = frame0[res[0]:res[1], res[2]:res[3], res[4]:res[5]]
    print(frame0.shape)
    # with h5py.File(fileName, "w") as f:
    #     f.create("ctData", data=frame0)
    code, name = readMaskData(findROIPath(fileName, maskDataPath), res)
    ctName = (fileName.split("/")[-1]).replace(".v", ".hdf5")
    maskName = ctName.replace("ctData", "maskData")
    with h5py.File(os.path.join(toDataPath, ctName), "w") as f:
        f.create_dataset("ctData", data=frame0, dtype=np.float32)
    with h5py.File(os.path.join(toDataPath, maskName), "w") as f:
        for key in code.keys():
            f.create_dataset(key, data=code[key], dtype=np.int8)
