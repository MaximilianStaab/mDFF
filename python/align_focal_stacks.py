#! /usr/bin/python3

import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

def __align_images(images, iterations=5000, termination_eps=1e-10):
    # Align all images to first image in list
    out_images = [images[0]]

    base_image_gray = cv2.cvtColor(images[0],cv2.COLOR_BGR2GRAY)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations,  termination_eps)
    for image in images[1:]:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC(base_image_gray,image_gray,warp_matrix, cv2.MOTION_AFFINE, criteria)
        # Warp image
        out_images += [cv2.warpAffine(image, warp_matrix, (images[0].shape[1],images[0].shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)]

    return out_images

def align_stack(stack_dir):
    stack = [cv2.imread(os.path.join(stack_dir,stack_image),cv2.IMREAD_UNCHANGED)
                        for stack_image in sorted(os.listdir(stack_dir)) 
                        if os.path.splitext(stack_image)[1] == ".jpg"]
    return __align_images(stack)

def __crop_stack(stack, crop):
    return [img[crop[2]:-crop[3], crop[0]:-crop[1]] for img in stack]

def __scale_stack(stack, scale_width):
    return [cv2.resize(img, (scale_width, int((scale_width/img.shape[1])*img.shape[0]))) for img in stack]

def align_stacks(stacks_dir, out_dir, threads=4, crop=None, scale_width=None):
    stack_folders = sorted([(root, os.path.join(out_dir, os.path.relpath(root,stacks_dir))) for root,_,files in os.walk(stacks_dir)
                            if files and all((os.path.splitext(f)[1] == ".jpg") for f in files)])


    if threads <= 0:
        for stack_folder, out_folder in tqdm(stack_folders):
            images_aligned = align_stack(stack_folder)
            #Crop images
            if crop is not None:
                images_aligned = __crop_stack(images_aligned, crop)
            if scale_width is not None:
                images_aligned = __scale_stack(images_aligned, scale_width)
            #Write imates to disk
            os.makedirs(out_folder, exist_ok=True)
            for idx, image_aligned in enumerate(images_aligned):
                cv2.imwrite(os.path.join(out_folder, str(idx) + ".jpg"), image_aligned)
    else:
        import multiprocessing
        with multiprocessing.Pool(processes=threads) as pool: 
            # Setup a list of processes that we want to run
            processes = [(pool.apply_async(align_stack, args=[stack_folder]), out_folder) for stack_folder, out_folder in stack_folders]
            for process, out_folder in tqdm(processes):
                images_aligned = process.get()
                #Crop images
                if crop is not None:
                    images_aligned = __crop_stack(images_aligned, crop)
                if scale_width is not None:
                    images_aligned = __scale_stack(images_aligned, scale_width)
                #Write imates to disk
                os.makedirs(out_folder, exist_ok=True)
                for idx, image_aligned in enumerate(images_aligned):
                    cv2.imwrite(os.path.join(out_folder, str(idx) + ".jpg"), image_aligned)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", help="input directory containing focal stacks", type=str)
    parser.add_argument("outdir", help="input directory containing focal stacks", type=str)
    parser.add_argument('--crop', help="tuple of crop values (left, right, top, bottom). If tuple contains None -> no crop", nargs=4, type=int, default=(348, 387, 245, 173))
    parser.add_argument('--scalewidth', help="with of the target image", type=int, default=552)
    parser.add_argument('--threads', help="number of threads", type=int, default=2)
    args = parser.parse_args()
    align_stacks(args.indir, args.outdir, threads=args.threads, crop=args.crop if all(args.crop) else None, scale_width=args.scalewidth) #crop left,right,top,bottom    scale h,w
    
