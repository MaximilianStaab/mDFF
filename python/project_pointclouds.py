#! /usr/bin/python3

import numpy as np
import quaternion
import os
import cv2
from collections import namedtuple
import argparse

#Define Intrinsics namedtuple
Intrinsics = namedtuple('Intrinsics', ['fx', 'fy', 'cx', 'cy', 'height', 'width', 'k1', 'k2', 'p1', 'p2', 'k3'])

def read_pointcloud(path, confidence=True):
    with open(path) as f:
        pointcloud = f.readlines()
    pointcloud = [[float(elem.strip()) for elem in line.split(",")] for line in pointcloud]
    if not confidence:
        pointcloud = [line[:-1] for line in pointcloud]
    return pointcloud

def project_pointcloud(pointcloud, depth_intrinsics, cam_rot, cam_trans, invert_x_axis=False):
    cam_rot = np.quaternion(*cam_rot)
    pointcloud_transformed = quaternion.rotate_vectors(cam_rot, pointcloud)
    pointcloud_transformed = pointcloud_transformed + np.asarray([cam_trans])
    #Create depth image
    depth_image = np.zeros((depth_intrinsics.height, depth_intrinsics.width))
    point_transformed_squared = np.square(pointcloud_transformed)
    point_transformed_ru = np.sqrt((point_transformed_squared[:,0] + point_transformed_squared[:,1]) / point_transformed_squared[:,2])
    point_transformed_rd = point_transformed_ru \
                            + depth_intrinsics.k1 * np.power(point_transformed_ru, 3) \
                            + depth_intrinsics.k2 * np.power(point_transformed_ru, 5) \
                            + depth_intrinsics.k3 * np.power(point_transformed_ru, 7)
    point_transformed_rd_ru = point_transformed_rd / point_transformed_ru
    pixels_x = (depth_intrinsics.fx \
                * point_transformed_rd_ru \
                * (pointcloud_transformed[:,0] / pointcloud_transformed[:,2]) \
                + depth_intrinsics.cx).astype(int)
    pixels_y = (depth_intrinsics.fy \
                * point_transformed_rd_ru \
                * (pointcloud_transformed[:,1] / pointcloud_transformed[:,2]) \
                + depth_intrinsics.cy).astype(int)

    #Flip x-axis
    if invert_x_axis:
        pixels_x = depth_image.shape[1] - pixels_x

    for pixel_x, pixel_y, point in zip(pixels_x, pixels_y, pointcloud):
        if pixel_x >= 0 and pixel_y >= 0 and pixel_x < depth_intrinsics.width and pixel_y < depth_intrinsics.height:
            depth_image[pixel_y,pixel_x] = point[2]
    return depth_image

def fuse_rgb_depth(depth_image, depth_intrinsics, color_intrinsics, goal_width, goal_height):
    width_scaling_factor = goal_width/color_intrinsics.width
    height_scaling_factor = goal_height/color_intrinsics.height
    scaling_dims = [int(width_scaling_factor*color_intrinsics.width), int(height_scaling_factor*color_intrinsics.height)]
    depth_image_scaled = np.zeros((min(scaling_dims), max(scaling_dims))) 

    x = np.arange(depth_image_scaled.shape[1])
    y = np.arange(depth_image_scaled.shape[0])
    xWorld = (x - color_intrinsics.cx*width_scaling_factor) / (color_intrinsics.fx*width_scaling_factor)
    yWorld = (y - color_intrinsics.cy*height_scaling_factor) / (color_intrinsics.fy*height_scaling_factor)

    xWorld_repeated = np.repeat(np.expand_dims(xWorld, axis=0),depth_image_scaled.shape[0], axis=0)
    yWorld_repeated = np.repeat(np.expand_dims(yWorld, axis=-1),depth_image_scaled.shape[1], axis=-1)

    ruDepth = np.sqrt(np.square(xWorld_repeated) + np.square(yWorld_repeated))
    rdDepth = ruDepth \
                + depth_intrinsics.k1 * np.power(ruDepth, 3) \
                + depth_intrinsics.k2 * np.power(ruDepth, 5) \
                + depth_intrinsics.k3 * np.power(ruDepth, 7)


    rdDepth_ruDepth = rdDepth / ruDepth
    xDepthSpaceMat = (xWorld_repeated*depth_intrinsics.fx*rdDepth_ruDepth + depth_intrinsics.cx).astype(int)
    yDepthSpaceMat = (yWorld_repeated*depth_intrinsics.fy*rdDepth_ruDepth + depth_intrinsics.cy).astype(int)

    for y in range(depth_image_scaled.shape[0]):
        for x in range(depth_image_scaled.shape[1]):
            xDepthSpace = xDepthSpaceMat[y,x]
            yDepthSpace = yDepthSpaceMat[y,x]
            if xDepthSpace >= 0 and yDepthSpace >= 0 and xDepthSpace < depth_intrinsics.width and yDepthSpace < depth_intrinsics.height:
                if depth_image_scaled.shape[0] > depth_image_scaled.shape[1]:
                    depth_image_scaled[x,y] = depth_image[xDepthSpace,yDepthSpace]
                else:
                    depth_image_scaled[y,x] = depth_image[yDepthSpace, xDepthSpace]
    return depth_image_scaled

def __process_point_cloud_folder(pointcloud_folder, depth_image_scaling_factor, goal_width, goal_height, color_intrinsics, depth_intrinsics, cam_rot, cam_trans, invert_x_axis):
    #Read point clouds from given folder
    pointclouds = [np.asarray(read_pointcloud(os.path.join(pointcloud_folder,pointcloud), confidence=False))
                        for pointcloud in os.listdir(pointcloud_folder) 
                        if os.path.splitext(pointcloud)[1] == ".txt" and 
                            pointcloud != "CameraTransform.txt" and 
                            pointcloud != "DepthIntrinsics.txt" and 
                            pointcloud != "ColorIntrinsics.txt"]
    #Project point clouds to 2D image plane
    depth_images = np.asarray([project_pointcloud(pointcloud, depth_intrinsics, cam_rot, cam_trans, invert_x_axis=invert_x_axis) for pointcloud in pointclouds])
    #Calculate median of depth images
    depth_image = np.median(depth_images, axis=0) * depth_image_scaling_factor
    depth_image_scaled = fuse_rgb_depth(depth_image, depth_intrinsics, color_intrinsics, goal_width, goal_height) 
    return depth_image_scaled.astype(np.uint16), depth_image.astype(np.uint16)


def generate_depth_maps(pointcloud_dir, out_dir, depth_filename, depth_image_scaling_factor, goal_width, goal_height, color_intrinsics, depth_intrinsics, cam_rot, cam_trans, invert_x_axis, color_depth=False, clip_depth=None):
    pointcloud_folders = sorted([(root, os.path.join(out_dir, os.path.relpath(root,pointcloud_dir))) for root,_,files in os.walk(pointcloud_dir)
                            if files and all((os.path.splitext(f)[1] == ".txt" or f == depth_filename) and 
                                                f != "CameraTransform.txt" and
                                                f != "DepthIntrinsics.txt" and 
                                                f != "ColorIntrinsics.txt" for f in files)])

    for pointcloud_folder in pointcloud_folders:
        #Create output folder
        os.makedirs(pointcloud_folder[1], exist_ok=True)
        #Project point cloud
        depth_image, depth_image_small = __process_point_cloud_folder(pointcloud_folder[0], depth_image_scaling_factor, goal_width, goal_height, color_intrinsics, depth_intrinsics, cam_rot, cam_trans, invert_x_axis)
        #Clip ground truth
        if clip_depth is not None:
            depth_image[depth_image < clip_depth[0]*depth_image_scaling_factor] = 0.0
            depth_image[depth_image > clip_depth[1]*depth_image_scaling_factor] = 0.0
        #Write output image
        print("Writing file: " + os.path.join(pointcloud_folder[1], depth_filename))
        if color_depth:
            depth_image = depth_image / depth_image_scaling_factor
            depth_image[depth_image > 8.0] = 0.0
            depth_image_jet = cv2.applyColorMap((((depth_image / np.max(depth_image)))*255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(pointcloud_folder[1], depth_filename), depth_image_jet)
        else:
            cv2.imwrite(os.path.join(pointcloud_folder[1], depth_filename), depth_image)
        depth_filename_split = os.path.splitext(depth_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", help="input directory containing focal stacks", type=str)
    parser.add_argument("outdir", help="input directory containing focal stacks", type=str)
    parser.add_argument('--camrot', help="quaternion representing camera rotation", nargs=4, type=int, default=(-0.00492409944960437,0.0032396232542505443,0.9999739917011292,0.004156201040729303))
    parser.add_argument('--camtrans', help="vector representing camera translation", nargs=3, type=int, default=(0.010849939503541267,5.906483056665815E-5,-0.0020920851586685237))
    parser.add_argument('--depthfilename', help="name of the generated depth file", type=str, default="depth.png")
    parser.add_argument('--depthscalingfactor', help="scaling factor to befor saving the depth values to 16 bit image", type=float, default=1e4)
    parser.add_argument('--goalres', help="resolution (width,height) of the generated depth image", nargs=2, type=int, default=(552, 310))
    parser.add_argument('--invertxaxis', help="invert x-axis of generated depth image", nargs=2, type=bool, default=True)
    parser.add_argument('--colordepth', action='store_true', help='Create a colored image from depth images')
    parser.add_argument('--clipdepth', type=float, nargs=2, default=None, help='Set groundtruth ouside given interval to invalid (defualt:None)')
    args = parser.parse_args()

    color_intrinsics = Intrinsics(1720.230053864719, 
                                    1721.937892726168, 
                                    950.1598302701242, 
                                    529.6482581071045, 
                                    1080, 
                                    1920, 
                                    0.07312925173521241, 
                                    -0.1189699348788761,
                                    -0.0758732942942454,
                                    0.0,
                                    0.0)
    depth_intrinsics = Intrinsics(216.026,
                                    216.026,
                                    111.951,
                                    85.5796,
                                    172,
                                    224,
                                    -0.155455,
                                    -1.28213,
                                    8.59888E-4,
                                    0.00116285,
                                    2.74844)

    generate_depth_maps(args.indir, args.outdir, args.depthfilename, args.depthscalingfactor, args.goalres[0], args.goalres[1], color_intrinsics, depth_intrinsics, args.camrot, args.camtrans, args.invertxaxis, color_depth=args.colordepth, clip_depth=args.clipdepth)
