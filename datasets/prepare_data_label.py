import os
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import box_utils
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
import numpy as np
import cv2
from PIL import Image
import mmengine
import tensorflow as tf
tf.config.set_visible_devices(tf.config.list_physical_devices("CPU"))
from tqdm import tqdm
import imageio 
import argparse


MAX_MAP = 75

UNKNOWN = 0
FRONT = 1
FRONT_LEFT = 2
FRONT_RIGHT = 3
SIDE_LEFT = 4
SIDE_RIGHT = 5
H = 1280
W = 1920

OPENCV2DATASET = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    )

calib_view_map = {
            'FRONT':0,
            'FRONT_LEFT':1,
            'FRONT_RIGHT':2,
            'SIDE_LEFT':3,
            'SIDE_RIGHT':4
        }
name_val_map = {
            'FRONT':FRONT,
            'FRONT_LEFT':FRONT_LEFT,
            'FRONT_RIGHT':FRONT_RIGHT,
            'SIDE_LEFT':SIDE_LEFT,
            'SIDE_RIGHT':SIDE_RIGHT
        }
cls_map = {
            1:0,
            2:1,
            4:2
            }
def get_intrinsic(calib):
    intrinsic = calib.intrinsic
    fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return intrinsic

def get_extrinsic(calib):
    extrinsic = np.array(calib.extrinsic.transform).reshape(4,4)
    extrinsic = extrinsic@OPENCV2DATASET
    return extrinsic

def _get_cam_calib(frame, cam_type):
    
    ego2world = np.array(frame.pose.transform).reshape(4,4)
    cam_calib = frame.context.camera_calibrations[calib_view_map[cam_type]]
    assert cam_calib.name==name_val_map[cam_type]
    cam_intrinsic = get_intrinsic(cam_calib)
    cam2ego = get_extrinsic(cam_calib)
    height = cam_calib.height
    width = cam_calib.width
    calib ={
        'intrinsic': cam_intrinsic,
        'ego2world': ego2world,
        'cam2ego': cam2ego,
        'height': height,
        'width': width,
    }
        
    return calib

def project_vehicle_to_image(vehicle_pose, calibration, points):
    """Projects from vehicle coordinate system to image with global shutter.

    Arguments:
      vehicle_pose: Vehicle pose transform from vehicle into world coordinate
        system.
      calibration: Camera calibration details (including intrinsics/extrinsics).
      points: Points to project of shape [N, 3] in vehicle coordinate system.

    Returns:
      Array of shape [N, 3], with the latter dimension composed of (u, v, ok).
    """
    # Transform points from vehicle to world coordinate system (can be
    # vectorized).
    pose_matrix = np.array(vehicle_pose.transform).reshape(4, 4)
    world_points = np.zeros_like(points)
    for i, point in enumerate(points):
        cx, cy, cz, _ = np.matmul(pose_matrix, [*point, 1])
        world_points[i] = (cx, cy, cz)

    # Populate camera image metadata. Velocity and latency stats are filled with
    # zeroes.
    extrinsic = tf.reshape(
        tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32), [4, 4]
    )
    intrinsic = tf.constant(list(calibration.intrinsic), dtype=tf.float32)
    metadata = tf.constant(
        [
            calibration.width,
            calibration.height,
            dataset_pb2.CameraCalibration.GLOBAL_SHUTTER,
        ],
        dtype=tf.int32,
    )
    camera_image_metadata = list(vehicle_pose.transform) + [0.0] * 10

    # Perform projection and return projected image coordinates (u, v, ok).
    return py_camera_model_ops.world_to_image(
        extrinsic, intrinsic, metadata, camera_image_metadata, world_points
    ).numpy(),world_points

def _get_label(frame, cam_type):
    boxes_3d = []
    corners = []
    types = []
    world_corners = []
    calib = frame.context.camera_calibrations[calib_view_map[cam_type]]
    
    for label in frame.laser_labels:
        ''' 
        enum Type {
          TYPE_UNKNOWN = 0;
          TYPE_VEHICLE = 1;
          TYPE_PEDESTRIAN = 2;
          TYPE_SIGN = 3;
          TYPE_CYCLIST = 4;
        }
        '''
        _type = label.type
        if _type not in cls_map:
            continue
        
        box = label.box
        if not box.ByteSize():
            continue  # Filter out labels that do not have a camera_synced_box.
        if  label.num_top_lidar_points_in_box< 10:
            continue  # Filter out likely occluded objects.
        
        box_coords = np.array(
                [
                    [
                        box.center_x,
                        box.center_y,
                        box.center_z,
                        box.length,
                        box.width,
                        box.height,
                        box.heading,
                    ]
                ]
            )
        # get ego box
        corner_3d = box_utils.get_upright_3d_box_corners(box_coords)[0].numpy()  # [8, 3]
        
        # Project box corners from vehicle coordinates onto the image.
        corner,world_corner = project_vehicle_to_image(
            frame.pose, calib, corner_3d
        )

        ok = corner[:,2]
        if not ok.any():
            continue
        
        
        
        # # waymo box format to nuScenes format
        corner = corner[[1,2,6,5,0,3,7,4],:2]
        world_corners.append(world_corner)
        corners.append(corner)
        boxes_3d.append(box_coords[0])
        types.append(_type)
        
    
    if len(boxes_3d) > 0:
        boxes_3d = np.stack(boxes_3d, axis=0).astype(np.float32)
        corners = np.stack(corners, axis=0).astype(np.float32)
        world_corners = np.stack(world_corners, axis=0).astype(np.float32)
        
    else:
        boxes_3d = np.zeros((0,7), dtype=np.float32)
        corners = np.zeros((0,8,2), dtype=np.float32)
        world_corners = np.zeros((0,8,3), dtype=np.float32)
        

    return boxes_3d,corners,world_corners,types

def _get_hdmap( frame,bev_hdmap):
    

    # if len(frame.map_features)>0:
    #     bev_hdmap = frame.map_feature
    #     assert 
    offset  = frame.map_pose_offset
    offset = np.array([offset.x,offset.y,offset.z,0])
    vectors = []
    
    for line in bev_hdmap:
        # get lines
        if line.HasField('road_edge'):
            vector = list(line.road_edge.polyline)
            _type = 'road_edge'
        elif line.HasField('road_line'):
            vector = list(line.road_line.polyline)
            _type = 'road_line'
        elif line.HasField('crosswalk'):
            vector = list(line.crosswalk.polygon)
            _type = 'crosswalk'
        else:
            continue
        
        # get points 
        pts = []
        for _pt in vector:
            pt = np.array([_pt.x,_pt.y,_pt.z,1])
            pt -= offset
            pts.append(pt)
        pts = np.stack(pts)
        
        # crosswalk only save the long side
        if _type == 'crosswalk':
            if pts.shape[0]== 4:
                dist = np.square(pts[1:]-pts[:-1]).sum(1)
                idx = np.argsort(dist)[-1]
                # pts = pts[idx:idx+2]
                if idx ==2:
                    idx = 0
                vectors.append((pts[idx:idx+2],_type))
                vectors.append((pts[[idx+2,(idx+3)%4]],_type))
            # assert idx ==1
        else:
            vectors.append((pts,_type))
    hdmap = vectors
    
    return hdmap

def prepare_data_label(args):
    split = args.split
    data_root = args.data_root
    save_root = args.save_root
    scene_ids = args.scene_ids

    data_root = os.path.join(data_root,split)
    save_root = os.path.join(save_root,split)
    os.makedirs(save_root,exist_ok=True)
    scenes = os.listdir(data_root)
    scenes.sort()
    
    cam_type = 'FRONT'
    for scene_id in tqdm(scene_ids):
        scene_id = int(scene_id)
        save_dir = os.path.join(save_root,str(scene_id).zfill(3))
        os.makedirs(save_root,exist_ok=True)
        scene = scenes[scene_id]
        data_dict = {
            'scene_id':scene_id
        }
        offsets = []
        calibs = []
        corners = []
        names = []
        hdmaps = []
        world_corners = []
        scene_data =  tf.data.TFRecordDataset(os.path.join(data_root,scene))
        for i,data in enumerate(scene_data):
            frame = dataset_pb2.Frame.FromString(bytearray(data.numpy()))
            if i == 0:
                bev_hdmap = frame.map_features

            hdmap = _get_hdmap(frame,bev_hdmap)
            calib = _get_cam_calib(frame,cam_type)
            boxes_3d,corner,world_corner,name = _get_label(frame,cam_type)
            offset  = frame.map_pose_offset
            offset = np.array([offset.x,offset.y,offset.z,0])
            
            offsets.append(offset)
            calibs.append(calib)
            corners.append(corner)
            names.append(name)
            hdmaps.append(hdmap)
            world_corners.append(world_corner)

           
        data_dict.update(
            {
                'offsets':offsets,
                'calibs':calibs,
                'corners':corners,
                'names':names,
                'hdmap':hdmaps,
                'world_corner':world_corners
            }
        )    
        mmengine.dump(data_dict,os.path.join(save_dir,'label.pkl'))

if __name__ == '__main__':
    
    def parse_args():
        parser = argparse.ArgumentParser(description="Prepare novel trajectories gt labels.")
        
        parser.add_argument('--data_root', type=str, default='/PATH/TO/YOUR/WAYMO/SOURCE/DATA', help='Data path')
        parser.add_argument('--save_root', type=str, default='./data/waymo/processed', help='save path')
        parser.add_argument('--scene_ids', type=str, nargs='+', default=['005'])
        parser.add_argument('--split', type=str, choices=['training','validation'],default='validation')
        return parser.parse_args()

    args = parse_args()
    prepare_data_label(args)


        




    

