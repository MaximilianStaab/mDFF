import argparse
import os
import cv2
import h5py
import random
import numpy as np
from tqdm import tqdm

def __create_crops(stack, disp, disp_clip, nonzerocent=0.8, patchsize=224):
    stride = int(patchsize/4)
    h, w = stack.shape[-3], stack.shape[-2]
    rows = np.arange(0, h+stride, stride)
    rows[rows+patchsize>h] = h-patchsize-1
    rows = np.unique(rows)
    cols = np.arange(0, w+stride, stride)
    cols[cols+patchsize>w] = w-patchsize-1
    cols = np.unique(cols)
    result_stk = []
    result_disp = []
    disp[disp < disp_clip[0]] = 0.0
    disp[disp > disp_clip[1]] = 0.0
    for row in rows:
        for col in cols:
            dispcrop = disp[row:row+patchsize, col:col+patchsize]
            if np.count_nonzero(dispcrop) >= (nonzerocent * patchsize**2):
                stkcrop = stack[:, row:row+patchsize, col:col+patchsize, :]
                result_stk += [stkcrop]
                result_disp += [dispcrop]
    return result_stk, result_disp

def pack_h5(stack_train_dir, disp_train_dir, stack_test_dir, disp_test_dir, disp_name, out_dir, disp_scale, disp_clip, train_val_split=0.8, patchsize=224):
    folders_train_val = sorted([os.path.normpath(os.path.relpath(root, disp_train_dir))
                            for root,dirs,files in os.walk(disp_train_dir) 
                            for f in files 
                            if f == disp_name 
                            and len(cv2.imread(os.path.join(root, f), cv2.IMREAD_UNCHANGED).shape) <= 2])
    folders_test = sorted([os.path.normpath(os.path.relpath(root, disp_test_dir))
                            for root,dirs,files in os.walk(disp_test_dir) 
                            for f in files 
                            if f == disp_name 
                            and len(cv2.imread(os.path.join(root, f), cv2.IMREAD_UNCHANGED).shape) <= 2])

    #Split training set into train and validation set
    split_index = int(train_val_split * len(folders_train_val))
    random.shuffle(folders_train_val)
    folders_train, folders_val = sorted(folders_train_val[:split_index]), sorted(folders_train_val[split_index:])
    #Read one training stack image to determine shape of stack images in trainig set
    train_sample_names = [os.path.join(os.path.join(stack_train_dir,folders_train_val[0]), f) for f 
                in os.listdir(os.path.join(stack_train_dir,folders_train_val[0])) 
                    if os.path.isfile(os.path.join(os.path.join(stack_train_dir,folders_train_val[0]), f)) 
                        and (os.path.splitext(f)[-1].lower() == ".png" 
                        or os.path.splitext(f)[-1].lower() == ".jpg")]
    #train_stack_shape = (len(folders_train),len(train_sample_names)) + cv2.cvtColor(cv2.imread(train_sample_names[0]), cv2.COLOR_BGR2RGB).astype(float).shape
    train_stack_shape = (1,len(train_sample_names)) + cv2.cvtColor(cv2.imread(train_sample_names[0]), cv2.COLOR_BGR2RGB).astype(float).shape
    #val_stack_shape = (len(folders_val),len(train_sample_names)) + cv2.cvtColor(cv2.imread(train_sample_names[0]), cv2.COLOR_BGR2RGB).astype(float).shape
    val_stack_shape = (1,len(train_sample_names)) + cv2.cvtColor(cv2.imread(train_sample_names[0]), cv2.COLOR_BGR2RGB).astype(float).shape
    
    #Read one training disp image to determine shape disp images in training set
    disp_sample_name = os.path.join(os.path.join(disp_train_dir,folders_train_val[0]), disp_name)
    #train_disp_shape = (len(folders_train),) + cv2.imread(disp_sample_name, cv2.IMREAD_ANYDEPTH).astype(float).shape
    train_disp_shape = (1,) + cv2.imread(disp_sample_name, cv2.IMREAD_ANYDEPTH).astype(float).shape
    #val_disp_shape = (len(folders_val),) + cv2.imread(disp_sample_name, cv2.IMREAD_ANYDEPTH).astype(float).shape
    val_disp_shape = (1,) + cv2.imread(disp_sample_name, cv2.IMREAD_ANYDEPTH).astype(float).shape

    #Read one test stack image to determine shape of stack images in test set
    test_sample_names = [os.path.join(os.path.join(stack_test_dir,folders_test[0]), f) for f 
                in os.listdir(os.path.join(stack_test_dir,folders_test[0])) 
                    if os.path.isfile(os.path.join(os.path.join(stack_test_dir,folders_test[0]), f)) 
                        and (os.path.splitext(f)[-1].lower() == ".png" 
                        or os.path.splitext(f)[-1].lower() == ".jpg")]
    test_stack_shape = (len(folders_test),len(test_sample_names)) + cv2.cvtColor(cv2.imread(test_sample_names[0]), cv2.COLOR_BGR2RGB).astype(float).shape
    
    #Read one test disp image to determine shape disp images in test set
    disp_sample_name = os.path.join(os.path.join(disp_test_dir,folders_test[0]), disp_name)
    test_disp_shape = (len(folders_test),) + cv2.imread(disp_sample_name, cv2.IMREAD_ANYDEPTH).astype(float).shape

    #Create hdf5 dataset
    hdf5 = h5py.File(out_dir, mode='w')
    hdf5.create_dataset("stack_train", shape=train_stack_shape, maxshape=(None,) + tuple(train_stack_shape[1:]))
    hdf5.create_dataset("disp_train", shape=train_disp_shape, maxshape=(None,) + tuple(train_disp_shape[1:]))
    hdf5.create_dataset("stack_val", shape=val_stack_shape, maxshape=(None,) + tuple(val_stack_shape[1:]))
    hdf5.create_dataset("disp_val", shape=val_disp_shape, maxshape=(None,) + tuple(val_disp_shape[1:]))
    hdf5.create_dataset("stack_test", test_stack_shape, np.float32)
    hdf5.create_dataset("disp_test", test_disp_shape, np.float32)

    #Add training set to h5 file
    print("Packing training set")
    for i, folder in enumerate(tqdm(folders_train)):
        #Determine stack image names
        stack_sample_names = [os.path.join(os.path.join(stack_train_dir,folder), f) for f 
                in os.listdir(os.path.join(stack_train_dir,folder)) 
                    if os.path.isfile(os.path.join(os.path.join(stack_train_dir,folder), f)) 
                        and (os.path.splitext(f)[-1].lower() == ".png" 
                        or os.path.splitext(f)[-1].lower() == ".jpg")]
        #Determine disp image name
        disp_sample_name = os.path.join(os.path.join(disp_train_dir,folder), disp_name)
        #Read stack images
        stack_samples = np.asarray([cv2.cvtColor(cv2.imread(stack_sample), cv2.COLOR_BGR2RGB).astype(float) for stack_sample in stack_sample_names])
        disp_sample = (cv2.imread(disp_sample_name, cv2.IMREAD_ANYDEPTH)*disp_scale).astype(float)
        #Crop stack samples
        stack_crops, disp_crops = __create_crops(stack_samples, disp_sample, disp_clip, patchsize=patchsize)
        #Add to dataset
        for stack_crop, disp_crop in zip(stack_crops, disp_crops):
            hdf5["stack_train"].resize((hdf5["stack_train"].len() + 1, stack_crop.shape[0], patchsize, patchsize, 3))
            hdf5["disp_train"].resize((hdf5["disp_train"].len() + 1, patchsize, patchsize))
            hdf5["stack_train"][-1] = stack_crop
            hdf5["disp_train"][-1] = disp_crop

    #Add validation set to h5 file
    print("Packing validation set")
    for i, folder in enumerate(tqdm(folders_val)):
        #Determine stack image names
        stack_sample_names = [os.path.join(os.path.join(stack_train_dir,folder), f) for f 
                in os.listdir(os.path.join(stack_train_dir,folder)) 
                    if os.path.isfile(os.path.join(os.path.join(stack_train_dir,folder), f)) 
                        and (os.path.splitext(f)[-1].lower() == ".png" 
                        or os.path.splitext(f)[-1].lower() == ".jpg")]
        
        #Determine disp image name
        disp_sample_name = os.path.join(os.path.join(disp_train_dir,folder), disp_name)
        #Read stack images
        stack_samples = np.asarray([cv2.cvtColor(cv2.imread(stack_sample), cv2.COLOR_BGR2RGB).astype(float) for stack_sample in stack_sample_names])
        disp_sample = (cv2.imread(disp_sample_name, cv2.IMREAD_ANYDEPTH)*disp_scale).astype(float)
        #Crop stack samples
        stack_crops, disp_crops = __create_crops(stack_samples, disp_sample, disp_clip, patchsize=patchsize)
        #Add to dataset
        for stack_crop, disp_crop in zip(stack_crops, disp_crops):
            hdf5["stack_val"].resize((hdf5["stack_val"].len() + 1, stack_crop.shape[0], patchsize, patchsize, 3))
            hdf5["disp_val"].resize((hdf5["disp_val"].len() + 1, patchsize, patchsize))
            hdf5["stack_val"][-1] = stack_crop
            hdf5["disp_val"][-1] = disp_crop

    #Add testing set to h5 file
    print("Packing test set")
    for i, folder in enumerate(tqdm(folders_test)):
        #Determine stack image names
        stack_sample_names = [os.path.join(os.path.join(stack_test_dir,folder), f) for f 
                in os.listdir(os.path.join(stack_test_dir,folder)) 
                    if os.path.isfile(os.path.join(os.path.join(stack_test_dir,folder), f)) 
                        and (os.path.splitext(f)[-1].lower() == ".png" 
                        or os.path.splitext(f)[-1].lower() == ".jpg")]
        #Determine disp image name
        disp_sample_name = os.path.join(os.path.join(disp_test_dir,folder), disp_name)
        #Read stack images
        stack_samples = np.asarray([cv2.cvtColor(cv2.imread(stack_sample), cv2.COLOR_BGR2RGB).astype(float) for stack_sample in stack_sample_names])
        disp_sample = (cv2.imread(disp_sample_name, cv2.IMREAD_ANYDEPTH)*disp_scale).astype(float)
        #Add to dataset
        hdf5["stack_test"][i] = stack_samples
        hdf5["disp_test"][i] = disp_sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stacktrain", help="input directory containing training focal stacks", type=str)
    parser.add_argument("disptrain", help="input directory containing training disparities", type=str)
    parser.add_argument("stacktest", help="input directory containing testing focal stacks", type=str)
    parser.add_argument("disptest", help="input directory containing testing disparities", type=str)
    parser.add_argument("outfile", help="h5 file to be written", type=str)
    parser.add_argument("--dispscale", help="scaling value to be applied on disparities", type=float, default=1e-4)
    parser.add_argument("--dispname", help="name of a disparity file", type=str, default="disp.png")
    parser.add_argument("--dispclip", help="Clipping bounds for disparity values. All values out of the given bounds will be considered invalid. Default:(0.1,4.0)", type=float, nargs=2, default=(0.1,4.0))
    parser.add_argument("--trainvalsplit", help="split ratio between training and validation set (default: 0.8", type=float, default=0.8)
    args = parser.parse_args()

    pack_h5(args.stacktrain, args.disptrain, args.stacktest, args.disptest, args.dispname, args.outfile, args.dispscale, args.dispclip, train_val_split=args.trainvalsplit)
