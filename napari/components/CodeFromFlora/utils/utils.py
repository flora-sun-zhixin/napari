# Flora Sun, WUSTL, 2021
import numpy as np
import os
import re
import shutil
import tensorflow as tf

def computeRF_recursive(filterArray):
    # compute the Receptive Field alone certain axis
    # filterArray is the 2D array, each row is the filter's metadata.
    # shallower(input) -> deeper(output)
    # col1 -- kernelsize along the axis
    # col2 -- stride along the axis
    # Example
    # filterList = np.array([[3, 2], [2, 1], [3, 2], [2, 1]])
    # print("The last [2, 1] CNN kernel's RF is: ", computeRF_recursive(filterList))
    import numpy as np
    rf = filterArray[-1, 0]
    for i in range(len(filterArray)-1, 0, -1):
        rf = rf * filterArray[i-1, 1] + filterArray[i-1, 0] - filterArray[i-1, 1]
    return rf

def computeRF_close_form(filterArray):
    # compute the Receptive Field alone certain axis
    # filterArray is the 2D array, each row is the filter's metadata.
    # shallower(input) -> deeper(output)
    # col1 -- kernelsize along the axis
    # col2 -- stride along the axis
    # Example
    # filterList = np.array([[3, 2], [2, 1], [3, 2], [2, 1]])
    # print("The last [2, 1] CNN kernel's RF is: ", computeRF_close_form(filterList))
    import numpy as np
    n = len(filterArray)
    strides = np.r_[1, filterArray[:-1, 1]]
    pre_prod = (np.tril(np.ones((n,n))) * strides).T    # generate a upper-triangle, prepare for the product
    pre_prod[pre_prod == 0] = 1
    prod = np.prod(pre_prod, axis=0)
    return int(prod.reshape(1, -1) @ (filterArray[:, 0] - 1).reshape(-1, 1) + 1)

def copytree(src=None, dst=None, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        if s.startswith("."):
            continue
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)

def patches_from_3Ddata(data3D,
                        p_size0=32 , p_overlap0=16,
                        p_size1=64 , p_overlap1=32,
                        p_size2=208, p_overlap2=104,
                        p_max0=100, p_max1=100, p_max2=100):
    """ split large images into small images
    Args:
        data3D: 3D matrix, (axis0, axis1, axis2, channels)
        p_size0: int, the size of the patches along axis0
        p_overlap0: int, smaller than p_size0, the overlap between patches along axis0
        p_max0: int, if the 3Dmatrix axis0 size exceed, then cut it, else, directly output the image
    Returns:
        np.array of pathes, containing either the original images or the cutted images. (patches, p_size/w, p_size/h, images_num)
    """
    axis0, axis1, axis2 = data3D.shape[:3]
    patches = []
    def generatePatchStartList(axis_size, p_size, p_overlap, p_max):
        if axis_size > p_max:
            axis_p = list(np.arange(0, axis_size-p_size, p_size-p_overlap, dtype=np.int)) # the starting point of each patch
            axis_p.append(axis_size - p_size) # make sure the right part gets in, even the overlap is bigger than others
        else:
            axis_p = [0]
            p_size = axis_size
        return axis_p, p_size

    axis0_p, p_size0 = generatePatchStartList(axis0, p_size0, p_overlap0, p_max0)
    axis1_p, p_size1 = generatePatchStartList(axis1, p_size1, p_overlap1, p_max1)
    axis2_p, p_size2 = generatePatchStartList(axis2, p_size2, p_overlap2, p_max2)

    for i in axis0_p:
        for j in axis1_p:
            for k in axis2_p:
                patches.append(data3D[i:i+p_size0, j:j+p_size1, k:k+p_size2, :])

    return np.array(patches, dtype=np.float32)

def patches_to_3Ddata(patches,
                      axis_size0=126,
                      axis_size1=172,
                      axis_size2=679,
                      channel = 1,
                      p_overlap0=16,
                      p_overlap1=32,
                      p_overlap2=104):
    """ put patches back together to a 3Ddata of shape (axis_size0, axis_size1, axis_size2), overlaped place got averaged.
    Args:
        patches: np.array of small 3D matrix plus channel, (axis0, axis1, axis2, channels)
        p_size0: int, the size of the patches along axis0
        p_overlap0: int, smaller than p_size0, the overlap between patches along axis0
        p_max0: int, if the 3Dmatrix axis0 size exceed, then cut it, else, directly output the image
    Returns:
        np.array, (axis_size0, axis_size1, axis_size2, channel)
    """
    p_size0, p_size1, p_size2 = patches[0,...].shape[:3]
    data3D = np.zeros((axis_size0, axis_size1, axis_size2, channel))
    data3D_count = np.zeros((axis_size0, axis_size1, axis_size2))
    def generatePatchStartList(axis_size, p_size, p_overlap):
        if axis_size > p_size:
            axis_p = list(np.arange(0, axis_size-p_size, p_size-p_overlap, dtype=np.int)) # the starting point of each patch
            axis_p.append(axis_size - p_size) # make sure the right part gets in, even the overlap is bigger than others
        else:
            axis_p = [0]
        return axis_p

    axis0_p = generatePatchStartList(axis_size0, p_size0, p_overlap0)
    axis1_p = generatePatchStartList(axis_size1, p_size1, p_overlap1)
    axis2_p = generatePatchStartList(axis_size2, p_size2, p_overlap2)

    startArray = np.array(np.meshgrid(axis0_p, axis1_p, axis2_p, indexing="ij")).transpose((1,2,3,0)).reshape(-1,3)
    for i in range(len(startArray)):
        s0, s1, s2 = startArray[i, :]
        data3D[s0:s0+p_size0, s1:s1+p_size1, s2:s2+p_size2, :] += patches[i]
        data3D_count[s0:s0+p_size0, s1:s1+p_size1, s2:s2+p_size2] += 1

    return data3D / data3D_count[..., np.newaxis]

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

# learning_rate generation functions ----------------------------------
def exp_decay(Ns, Ne, epochs):
    lamda = - (1/epochs) * np.log(Ne/Ns)
    return np.array(Ns*np.exp(-lamda*np.arange(epochs)))

def linear_decay(Ns, Ne, epochs):
    decrease = (Ns - Ne) / epochs
    return np.array(Ns - decrease * np.arange(epochs + 1))
        
def step_warper(lrArray, times, toLeft=True):
    step = (lrArray.shape[0] - 1) // times
    if toLeft:
        stepLrArray = lrArray[0::step]
    else: # to right
        stepLrArray = np.r_[lrArray[step::step], lrArray[-1]]
    stepLrArray = np.repeat(stepLrArray, step)
    return stepLrArray[:len(lrArray)]

def normalize(image):
    minValue = tf.math.reduce_min(image, axis=(-1, -2, -3), keepdims=True)
    maxValue = tf.math.reduce_max(image, axis=(-1, -2, -3), keepdims=True)
    return (image - minValue) / (maxValue - minValue + 1e-7)

def standardize(image):
    mean = tf.reduce_mean(image, axis=(-1, -2, -3), keepdims=True)
    std = tf.math.reduce_std(image, axis=(-1, -2, -3), keepdims=True)
    return (image - mean) / (std + 1e-7)

def putWeightOnBound(mask, boundWeight):
    """
    if either up, down, right or left is not the same as middle, consider it as bound,
    mask is (batch, height, width, channel)
    Args:
        mask: the one hot mask in shape of (batch, height, width, channel)
        boundWeight: the weight we want to put on the bound with everywhere else is 1
    
    Returns:
    a matrix in the same shape of mask, with bounds is boundWeight, and everywhere else is 1
    """
    for i in range(1, mask.ndim - 1):
        mask = mask + tf.roll(mask, -1, axis=i) + tf.roll(mask, 1, axis=i)
    return tf.ones(mask.shape) + tf.where(mask * (1 + 2 * (mask.ndim - 2) - mask) > 0, 1., 0.) * (boundWeight - 1)

def splitDataset(dataList, portion):
    """
    a function that split the dataList by the given portion
    dataList: a ndarray
    portion: a float in [0, 1]
    """
    print("number of split the dataset: ", int(len(dataList) * portion))
    part1 = np.random.choice(dataList, int(len(dataList) * portion), replace=False)
    part2 = np.setdiff1d(dataList, part1)
    return part1, part2
