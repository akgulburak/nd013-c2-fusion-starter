# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Process the point-cloud and prepare it for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import cv2
import numpy as np
import torch

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2

# object detection tools and helper functions
import misc.objdet_tools as tools

import zlib
import open3d as o3d

FRAME_SKIPPED = False
VISUALIZER = None

def load_frame_based_on_lidar_name(frame, lidar_name):
    for ith_image_in_frame in frame.lasers:
        if lidar_name == ith_image_in_frame.name:
            return ith_image_in_frame

def key_callback_destroy(visualizer):
    visualizer.destroy_window()
    exit()

def key_callback_skip(visualizer):
    global FRAME_SKIPPED
    FRAME_SKIPPED = True
    visualizer.close()

# visualize lidar point-cloud
def show_pcl(pcl):
    global VISUALIZER, FRAME_SKIPPED
    ####### ID_S1_EX2 START #######     
    #######
    print("student task ID_S1_EX2")

    # step 1 : initialize open3d with key callback and create window
    
    # step 2 : create instance of open3d point-cloud class

    # step 3 : set points in pcd instance by converting the point-cloud into 3d vectors (using open3d function Vector3dVector)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pcl[:, :3])
    if not FRAME_SKIPPED:
        VISUALIZER = o3d.visualization.VisualizerWithKeyCallback()
        VISUALIZER.register_key_callback(113, key_callback_destroy)
        VISUALIZER.register_key_callback(262, key_callback_skip)
        VISUALIZER.create_window()
        VISUALIZER.add_geometry(point_cloud)
    else:
        VISUALIZER.update_geometry(point_cloud)
        FRAME_SKIPPED = False

    # step 4 : for the first frame, add the pcd instance to visualization using add_geometry; for all other frames, use update_geometry instead
    
    VISUALIZER.poll_events()
    VISUALIZER.run()
    # step 5 : visualize point cloud and keep window open until right-arrow is pressed (key-code 262)

    #######
    ####### ID_S1_EX2 END #######     

def preprocess_range_scale_of_lidar_data(lidar_data):
    print(np.min(lidar_data))
    lidar_data[lidar_data<0]=0
    print(np.max(lidar_data))
    print(lidar_data[lidar_data!=0])

def map_range_image_to_8_bit(range_image):
    range_image[range_image<0]=0
    minimum_value = np.min(range_image)
    maximum_value = np.max(range_image)
    normalized_range_image = (range_image-minimum_value)/(maximum_value-minimum_value)
    bit_8_range_image = (normalized_range_image*255).astype(np.uint8)
    return bit_8_range_image

def map_intensity_image_to_8_bit(intensity_image):
    intensity_image[intensity_image<0]=0
    first_percentile = np.percentile(intensity_image, 1)
    last_percentile = np.percentile(intensity_image, 99)
    intensity_image_scaled = (intensity_image-first_percentile)/(last_percentile-first_percentile)
    bit_8_intensity_image = (intensity_image_scaled*255).astype(np.uint8)
    return bit_8_intensity_image

# visualize range image
def show_range_image(frame, lidar_name):

    ####### ID_S1_EX1 START #######     
    #######
    print("student task ID_S1_EX1")

    # step 1 : extract lidar data and range image for the roof-mounted lidar
    loaded_frame = load_frame_based_on_lidar_name(frame, lidar_name)
    loaded_range_image = loaded_frame.ri_return1.range_image_compressed
    ri = dataset_pb2.MatrixFloat()
    ri.ParseFromString(zlib.decompress(loaded_range_image))
    reshaped_decoded_image = np.array(ri.data).reshape(ri.shape.dims)
    range_image = reshaped_decoded_image[:, :, 0]
    intensity_image = reshaped_decoded_image[:, :, 1]
    bit_8_range_image = map_range_image_to_8_bit(range_image)
    bit_8_intensity_image = map_intensity_image_to_8_bit(intensity_image)
    img_range_intensity = np.concatenate((bit_8_range_image, bit_8_intensity_image), axis=0)
    # step 2 : extract the range and the intensity channel from the range image
    
    # step 3 : set values <0 to zero
    
    # step 4 : map the range channel onto n 8-bit scale and make sure that the full range of values is appropriately considered
    
    # step 5 : map the intensity channel onto an 8-bit scale and normalize with the difference between the 1- and 99-percentile to mitigate the influence of outliers

    # step 6 : stack the range and intensity image vertically using np.vstack and convert the result to an unsigned 8-bit integer
    
    #img_range_intensity = [] # remove after implementing all steps
    #######
    ####### ID_S1_EX1 END #######     
    
    return img_range_intensity


# create birds-eye view of lidar data
def bev_from_pcl(lidar_pcl, configs):

    # remove lidar points outside detection area and with too low reflectivity
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask]
    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]

    # convert sensor coordinates to bev-map coordinates (center is bottom-middle)
    ####### ID_S2_EX1 START #######
    #######
    print("student task ID_S2_EX1")

    ## step 1 :  compute bev-map discretization by dividing x-range by the bev-image height (see configs)
    x_range = configs.lim_x[1] - configs.lim_x[0]
    x_discretize_multiplier = x_range / configs.bev_width
    ## step 2 : create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates    
    lidar_pcl_cpy = lidar_pcl.copy()
    x_coordinates = lidar_pcl_cpy[:, 0]
    transformed_x_coordinates = x_coordinates / x_discretize_multiplier
    # step 3 : perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur
    y_range = configs.lim_y[1] - configs.lim_y[0]
    y_discretize_multiplier = y_range / configs.bev_width
    y_coordinates = lidar_pcl_cpy[:, 1] - lidar_pcl_cpy[:, 1].min()
    transformed_y_coordinates = y_coordinates / y_discretize_multiplier

    # step 4 : visualize point-cloud using the function show_pcl from a previous task
    transformed_coordinates = np.stack([transformed_x_coordinates, transformed_y_coordinates, lidar_pcl[:, 2]], axis=1)
    show_pcl(transformed_coordinates)
    #######
    ####### ID_S2_EX1 END #######     
    
    
    # Compute intensity layer of the BEV map
    ####### ID_S2_EX2 START #######     
    #######
    print("student task ID_S2_EX2")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    intensity_map = np.zeros((configs.bev_height, configs.bev_width))
    # step 2 : re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -z (use numpy.lexsort)
    lidar_pcl_cpy[lidar_pcl_cpy[:, 3]>1.0, 3] = 1.0
    sorted_indices = np.lexsort((lidar_pcl_cpy[:, 0], lidar_pcl_cpy[:, 1], -lidar_pcl_cpy[:, 2]))
    sorted_lidar = lidar_pcl_cpy[sorted_indices]
    ## step 3 : extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    ##          also, store the number of points per x,y-cell in a variable named "counts" for use in the next task
    _, unique_indices = np.unique(sorted_lidar[:, :2], axis=0, return_index=True)
    lidar_pcl_top = lidar_pcl_cpy[unique_indices]
    ## step 4 : assign the intensity value of each unique entry in lidar_top_pcl to the intensity map 
    ##          make sure that the intensity is scaled in such a way that objects of interest (e.g. vehicles) are clearly visible    
    ##          also, make sure that the influence of outliers is mitigated by normalizing intensity on the difference between the max. and min. value within the point cloud
    intensity_map[np.int_(transformed_x_coordinates), np.int_(transformed_y_coordinates)] = (lidar_pcl_top[:, 3] - np.min(lidar_pcl_top[:, 3])) / (np.max(lidar_pcl_top[:, 3])-np.min(lidar_pcl_top[:, 3]))
    ## step 5 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    cv2.imwrite("intensity_map.png", intensity_map*255)
    #######
    ####### ID_S2_EX2 END ####### 


    # Compute height layer of the BEV map
    ####### ID_S2_EX3 START #######     
    #######
    print("student task ID_S2_EX3")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    height_map = np.zeros_like(lidar_pcl_cpy)

    ## step 2 : assign the height value of each unique entry in lidar_top_pcl to the height map 
    ##          make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    ##          use the lidar_pcl_top data structure from the previous task to access the pixels of the height_map
    height_map = np.zeros((configs.bev_height, configs.bev_width))
    height_map[np.int_(transformed_x_coordinates), np.int_(transformed_y_coordinates)] = lidar_pcl_top[:, 2] / float(np.abs(configs.lim_z[1] - configs.lim_z[0]))
    ## step 3 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    cv2.imwrite("height_map.png", height_map*255)
    
    #######
    ####### ID_S2_EX3 END #######       

    # TODO remove after implementing all of the above steps
    #lidar_pcl_cpy = []
    #lidar_pcl_top = []
    #height_map = []
    #intensity_map = []

    # Compute density layer of the BEV map
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    _, _, counts = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64)) 
    density_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = normalizedCounts
        
    # assemble 3-channel bev-map from individual maps
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]  # r_map
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]  # g_map
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]  # b_map

    # expand dimension of bev_map before converting into a tensor
    s1, s2, s3 = bev_map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = bev_map

    bev_maps = torch.from_numpy(bev_maps)  # create tensor from birds-eye view
    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
    return input_bev_maps
