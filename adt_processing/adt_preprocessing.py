import numpy as np
import os
os.nice(5)
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
import math
from math import tan
import random
from scipy.linalg import pinv
import projectaria_tools.core.mps as mps
import shutil
import json
from PIL import Image
from utils import remake_dir
import pandas as pd
import pylab as p
from IPython.display import display
import time


from projectaria_tools import utils
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core import calibration
from projectaria_tools.projects.adt import (
   AriaDigitalTwinDataProvider,
   AriaDigitalTwinSkeletonProvider,
   AriaDigitalTwinDataPathsProvider,
   bbox3d_to_line_coordinates,
   bbox2d_to_image_coordinates,
   utils as adt_utils,
   Aria3dPose
)


dataset_path = '/datasets/public/zhiming_datasets/adt/'
dataset_processed_path = '/scratch/hu/pose_forecast/adt_hoigaze/'

remake_dir(dataset_processed_path)
remake_dir(dataset_processed_path + "train/")
remake_dir(dataset_processed_path + "test/")
dataset_info = pd.read_csv('adt.csv')
object_num = 5 # number of extracted dynamic objects that are closest to the left or right hands


for i, seq in enumerate(dataset_info['sequence_name']):        
    action = dataset_info['action'][i]
    print("\nprocessing {}th seq: {}, action: {}...".format(i+1, seq, action))
    seq_path = dataset_path + seq + '/'
    if dataset_info['training'][i] == 1:
        save_path = dataset_processed_path + 'train/' + seq + '_'                    
    if dataset_info['training'][i] == 0:
        save_path = dataset_processed_path + 'test/' + seq + '_'        
        
    paths_provider = AriaDigitalTwinDataPathsProvider(seq_path)
    all_device_serials = paths_provider.get_device_serial_numbers()
    selected_device_number = 0
    data_paths = paths_provider.get_datapaths_by_device_num(selected_device_number)
    print("loading ground truth data...")
    gt_provider = AriaDigitalTwinDataProvider(data_paths)
    print("loading ground truth data done")
    
    stream_id = StreamId("214-1")
    img_timestamps_ns = gt_provider.get_aria_device_capture_timestamps_ns(stream_id)
    frame_num = len(img_timestamps_ns)
    print("There are {} frames".format(frame_num))

    # get all available skeletons in a sequence
    skeleton_ids = gt_provider.get_skeleton_ids()
    skeleton_info = gt_provider.get_instance_info_by_id(skeleton_ids[0])
    print("skeleton ", skeleton_info.name, " wears ", skeleton_info.associated_device_serial)
    
    useful_frames = []
    gaze_data = np.zeros((frame_num, 6)) # gaze_direction (3) + gaze_2d (2) + frame_id (1)
    head_data = np.zeros((frame_num, 6)) # head_direction (3) + head_translation (3)
    hand_data = np.zeros((frame_num, 6)) # left_hand_translation (3) + right_hand_translation (3)
    hand_joint_data = np.zeros((frame_num, 92)) # left_hand (15*3) + right_hand (15*3) + attended_hand_gt + attended_hand_baseline (closest_hand)
    object_all_data = []
    object_bbx_all_data = []
    object_center_all_data = []    
    
    local_time = time.asctime(time.localtime(time.time()))
    print('\nProcessing starts at ' + local_time)    
    for j in range(frame_num):
        timestamps_ns = img_timestamps_ns[j]
        
        skeleton_with_dt = gt_provider.get_skeleton_by_timestamp_ns(timestamps_ns, skeleton_ids[0])
        assert skeleton_with_dt.is_valid(), "skeleton is not valid"
        
        skeleton = skeleton_with_dt.data()
        head_translation_id = [4]
        hand_translation_id = [8, 27]
        hand_joints_id = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]               
        hand_translation = np.array(skeleton.joints)[hand_translation_id, :].reshape(2*3)
        head_translation = np.array(skeleton.joints)[head_translation_id, :].reshape(1*3)
        hand_joints = np.array(skeleton.joints)[hand_joints_id, :].reshape(30*3)
        hand_data[j] = hand_translation
        hand_joint_data[j, :90] = hand_joints
        left_hand_joints = hand_joints[:45].reshape(15, 3)
        left_hand_center = np.mean(left_hand_joints, axis=0)
        right_hand_joints = hand_joints[45:].reshape(15, 3)
        right_hand_center = np.mean(right_hand_joints, axis=0)
        
        # get the Aria pose
        aria3dpose_with_dt = gt_provider.get_aria_3d_pose_by_timestamp_ns(timestamps_ns)
        if not aria3dpose_with_dt.is_valid():
            print("aria 3d pose is not available")
        aria3dpose = aria3dpose_with_dt.data()        
        transform_scene_device = aria3dpose.transform_scene_device.matrix()

        # get projection function
        cam_calibration = gt_provider.get_aria_camera_calibration(stream_id)
        assert cam_calibration is not None, "no camera calibration"

        eye_gaze_with_dt = gt_provider.get_eyegaze_by_timestamp_ns(timestamps_ns)
        assert eye_gaze_with_dt.is_valid(), "Eye gaze not available"
        
        # Project the gaze center in CPF frame into camera sensor plane, with multiplication performed in homogenous coordinates
        eye_gaze = eye_gaze_with_dt.data()
        gaze_center_in_cpf = np.array([tan(eye_gaze.yaw), tan(eye_gaze.pitch), 1.0], dtype=np.float64) * eye_gaze.depth        
        head_center_in_cpf = np.array([0.0, 0.0, 1.0], dtype=np.float64)        
        transform_cpf_sensor = gt_provider.raw_data_provider_ptr().get_device_calibration().get_transform_cpf_sensor(cam_calibration.get_label())
        gaze_center_in_camera = transform_cpf_sensor.inverse().matrix() @ np.hstack((gaze_center_in_cpf, 1)).T
        gaze_center_in_camera = gaze_center_in_camera[:3] / gaze_center_in_camera[3:]               
        gaze_center_in_pixels = cam_calibration.project(gaze_center_in_camera)
        head_center_in_camera = transform_cpf_sensor.inverse().matrix() @ np.hstack((head_center_in_cpf, 0)).T
        head_center_in_camera = head_center_in_camera[:3]        
        
        extrinsic_matrix = cam_calibration.get_transform_device_camera().matrix()
        gaze_center_in_device = (extrinsic_matrix @ np.hstack((gaze_center_in_camera, 1)))[0:3]
        gaze_center_in_scene = (transform_scene_device @ np.hstack((gaze_center_in_device, 1)))[0:3]
        head_center_in_device = (extrinsic_matrix @ np.hstack((head_center_in_camera, 0)))[0:3]
        head_center_in_scene = (transform_scene_device @ np.hstack((head_center_in_device, 0)))[0:3]        
        
        gaze_direction = gaze_center_in_scene - head_translation
        if np.linalg.norm(gaze_direction) == 0: # invalid data that will be filtered
            gaze_direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)            
        else:
            gaze_direction = [x / np.linalg.norm(gaze_direction) for x in gaze_direction]
        head_direction = head_center_in_scene        
        head_direction = [x / np.linalg.norm(head_direction) for x in head_direction]        
        head_data[j, 0:3] = head_direction
        head_data[j, 3:6] = head_translation
        
        left_hand_direction = left_hand_center - head_translation
        left_hand_direction = np.array([x / np.linalg.norm(left_hand_direction) for x in left_hand_direction]) 
        left_hand_distance_to_gaze = np.arccos(np.sum(gaze_direction*left_hand_direction))
        right_hand_direction = right_hand_center - head_translation
        right_hand_direction = np.array([x / np.linalg.norm(right_hand_direction) for x in right_hand_direction]) 
        right_hand_distance_to_gaze = np.arccos(np.sum(gaze_direction*right_hand_direction))
        if left_hand_distance_to_gaze < right_hand_distance_to_gaze:
            hand_joint_data[j, 90:91] = 0
        else:
            hand_joint_data[j, 90:91] = 1
        
        if gaze_center_in_pixels is not None:
            x_pixel = gaze_center_in_pixels[1]
            y_pixel = gaze_center_in_pixels[0]
            gaze_center_in_pixels[0] = x_pixel
            gaze_center_in_pixels[1] = y_pixel
                            
            useful_frames.append(j)
            gaze_2d = np.divide(gaze_center_in_pixels, cam_calibration.get_image_size())

            gaze_data[j, 0:3] = gaze_direction
            gaze_data[j, 3:5] = gaze_2d
            gaze_data[j, 5:6] = j
                
        # get the objects
        bbox3d_with_dt = gt_provider.get_object_3d_boundingboxes_by_timestamp_ns(timestamps_ns)
        assert bbox3d_with_dt.is_valid(), "3D bounding box is not available"
        bbox3d_all = bbox3d_with_dt.data()
        
        object_all = []
        object_bbx_all = []
        object_center_all = []
        
        for obj_id in bbox3d_all:
            bbox3d = bbox3d_all[obj_id]
            aabb = bbox3d.aabb
            aabb_coords = bbox3d_to_line_coordinates(aabb)
            obb = np.zeros(shape=(len(aabb_coords), 3))
            for k in range(0, len(aabb_coords)):
                aabb_pt = aabb_coords[k]
                aabb_pt_homo = np.append(aabb_pt, [1])
                obb_pt = (bbox3d.transform_scene_object.matrix() @ aabb_pt_homo)[0:3]
                obb[k] = obb_pt
            motion_type = gt_provider.get_instance_info_by_id(obj_id).motion_type
            if(str(motion_type) == 'MotionType.DYNAMIC'):
                object_all.append(obb)
                bbx_idx = [0, 1, 2, 3, 5, 6, 7, 8]
                obb_bbx = obb[bbx_idx, :]
                object_bbx_all.append(obb_bbx)
                obb_center = np.mean(obb_bbx, axis=0)
                object_center_all.append(obb_center)
                
        object_all_data.append(object_all)
        object_bbx_all_data.append(object_bbx_all)
        object_center_all_data.append(object_center_all)
        
    gaze_data = gaze_data[useful_frames, :] # useful_frames are actually continuous
    head_data = head_data[useful_frames, :]
    hand_data = hand_data[useful_frames, :]
    hand_joint_data = hand_joint_data[useful_frames, :]
    
    object_all_data = np.array(object_all_data)
    object_all_data = object_all_data[useful_frames, :, :, :]
    #print("Objects shape: {}".format(object_all_data.shape))    
    object_bbx_all_data = np.array(object_bbx_all_data)
    object_bbx_all_data = object_bbx_all_data[useful_frames, :, :, :]        
    object_center_all_data = np.array(object_center_all_data)
    object_center_all_data = object_center_all_data[useful_frames, :, :]
    
    # extract the closest objects to the left or right hands
    useful_frames_num = len(useful_frames)
    print("There are {} useful frames".format(useful_frames_num))
    object_num_all = object_all_data.shape[1]    
    object_left_hand_data = np.zeros((useful_frames_num, object_num, 16, 3))    
    object_bbx_left_hand_data = np.zeros((useful_frames_num, object_num, 8, 3))    
    object_distance_to_left_hand = np.zeros((useful_frames_num, object_num_all))    
    object_right_hand_data = np.zeros((useful_frames_num, object_num, 16, 3))
    object_bbx_right_hand_data = np.zeros((useful_frames_num, object_num, 8, 3))
    object_distance_to_right_hand = np.zeros((useful_frames_num, object_num_all))
    
    for j in range(useful_frames_num):
        left_hand_joints = hand_joint_data[j, :45].reshape(15, 3)
        right_hand_joints = hand_joint_data[j, 45:90].reshape(15, 3)
        for k in range(object_num_all):                    
            object_pos = object_center_all_data[j, k, :]
            object_distance_to_left_hand[j, k] = np.mean(np.linalg.norm(left_hand_joints-object_pos, axis=1))
            object_distance_to_right_hand[j, k] = np.mean(np.linalg.norm(right_hand_joints-object_pos, axis=1))
    
    for j in range(useful_frames_num):        
        distance_to_left_hand = object_distance_to_left_hand[j, :]
        distance_to_left_hand_min = np.min(distance_to_left_hand)
        distance_to_right_hand = object_distance_to_right_hand[j, :]
        distance_to_right_hand_min = np.min(distance_to_right_hand)
        if distance_to_left_hand_min < distance_to_right_hand_min:
            hand_joint_data[j, 91:92] = 0            
        else:
            hand_joint_data[j, 91:92] = 1
            
        left_hand_index = np.argsort(distance_to_left_hand)
        right_hand_index = np.argsort(distance_to_right_hand)
        for k in range(object_num):        
            object_left_hand_data[j, k] = object_all_data[j, left_hand_index[k]]
            object_bbx_left_hand_data[j, k] = object_bbx_all_data[j, left_hand_index[k]]
            object_right_hand_data[j, k] = object_all_data[j, right_hand_index[k]]            
            object_bbx_right_hand_data[j, k] = object_bbx_all_data[j, right_hand_index[k]]
                    
    gaze_path = save_path + 'gaze.npy'
    head_path = save_path + 'head.npy'        
    hand_path = save_path + 'hand.npy'
    hand_joint_path = save_path + 'handjoints.npy'
    object_left_hand_path = save_path + 'object_left.npy'
    object_bbx_left_hand_path = save_path + 'object_bbxleft.npy'    
    object_right_hand_path = save_path + 'object_right.npy'    
    object_bbx_right_hand_path = save_path + 'object_bbxright.npy'
    
    np.save(gaze_path, gaze_data)
    np.save(head_path, head_data)    
    np.save(hand_path, hand_data)
    np.save(hand_joint_path, hand_joint_data)
    np.save(object_left_hand_path, object_left_hand_data)
    np.save(object_bbx_left_hand_path, object_bbx_left_hand_data)
    np.save(object_right_hand_path, object_right_hand_data)
    np.save(object_bbx_right_hand_path, object_bbx_right_hand_data)
    
    local_time = time.asctime(time.localtime(time.time()))
    print('\nProcessing ends at ' + local_time)