import mmengine
import os
import numpy as np
import cv2
import imageio

def prepare_novel_gt_label(args):
    exp_name = args.exp_name
    exp_root = args.exp_root
    data_root = args.data_root
    scene_ids = args.scene_ids
    save_root = args.save_root
    

    start_idx_map = {
        '005':120
    }

    exp_root = os.path.join(exp_root,exp_name)
    if scene_ids is None:
        scene_ids = os.listdir(exp_root)
        scene_ids.sort()
    save_dir = os.path.join(save_root,'novel_gt_label')
    for scene in scene_ids:
        img_dir = os.path.join(exp_root,scene,'videos_eval','images','novel')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
            video_dir = os.path.join(exp_root,scene,'videos_eval','novel_30000')
            video_names = os.listdir(video_dir)
            for video_name in video_names:
                video = imageio.mimread(os.path.join(video_dir,video_name),memtest=False)
                novel_save_dir= os.path.join(img_dir,video_name[:-4])
                os.makedirs(novel_save_dir,exist_ok=True)
                for i,img in enumerate(video):
                    save_path = os.path.join(novel_save_dir,str(i).zfill(3)+'.png')
                    imageio.imwrite(save_path,img)
        
        novel_trajs = os.listdir(img_dir)
        novel_trajs.sort()
        start_idx = start_idx_map[scene]
        for novel_traj in novel_trajs:
            num_frames = len(os.listdir(os.path.join(img_dir,novel_traj)))
            data_path = os.path.join(data_root,scene,'label.pkl')
            data_dict = mmengine.load(data_path)
            corners_all_frame = data_dict['world_corner']
            labels_all_frame = data_dict['names']
            calibs = data_dict['calibs']
            imgs =[ ]
            for j in range(num_frames):
                i = start_idx+j
                save_path =os.path.join(save_dir,scene,novel_traj,f'{j:03d}.txt')
                if os.path.exists(save_path):
                    continue
                calib =calibs[i]
                cur_ego2world = calib['ego2world']
                cam2cur_ego = calib['cam2ego']
                intrinsic = calib['intrinsic']
                if j==0:
                    eog_02world = cur_ego2world
                    world2ego_0 = np.linalg.inv(eog_02world)#.astype(np.float128)
                cur_ego2ego_0 = world2ego_0@cur_ego2world
                if 'change_lane_left' in novel_traj:
                    cur_ego2ego_0[1,3]+=0.1*j
                elif 'change_lane_right' in novel_traj:
                    cur_ego2ego_0[1,3]-=0.1*j
                elif 'acc' in novel_traj:
                    cur_ego2ego_0[0,3]+=0.1*j
                elif 'dec' in novel_traj:
                    cur_ego2ego_0[0,3]-=0.1*j
                else:
                    assert False, print('Please check the type of novel trajectory')
                ref_c2w=eog_02world@cur_ego2ego_0@cam2cur_ego
                corners = corners_all_frame[i]
                labels = labels_all_frame[i]
                
                img= get_box_canva(corners,labels,ref_c2w,intrinsic,320,480,save_path)
                imgs.append(img)
            
def get_box_canva(corners,labels,c2w,intrinsic,height,width,save_path):
    world2cam = np.linalg.inv(c2w).astype(np.float128)
    scale = width/1920.
    # scale = 1
    new_corners = []
    new_labels = []
    # depths = []
    corners = np.concatenate([corners,np.ones((corners.shape[0],corners.shape[1],1))],axis=-1)
    for i,corner in enumerate(corners):
        if labels[i] != 1:
            continue
        corner = world2cam@corner.T
        depth = corner[2]
        corner,_depth = view_points_depth(corner,intrinsic,normalize=True)
        
        ok = _depth>0
        if not ok.any():
            continue
        corner = corner[:2,[1,2,6,5,0,3,7,4]].T
        corner = np.concatenate([corner,depth[:,None]],axis=1)
        new_corners.append(corner*scale)
        new_labels.append(labels[i])

    if len(new_corners)==0:
        new_corners = np.zeros((0,8,3), dtype=np.float32)
    else:
        new_corners = np.stack(new_corners, axis=0).astype(np.float32)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create the directory if it doesn't exist
    # dynamic_mask = np.zeros((height,width,1), dtype=np.uint8)  # 创建一个全黑的画布

    with open(save_path, 'w') as f:
        for corners in new_corners:
            # Get the min and max coordinates for the bounding box
            x_min, y_min, _ = corners.min(axis=0)
            x_max, y_max, _ = corners.max(axis=0)
            x_min,y_min = int(x_min),int(y_min)
            x_max,y_max = int(x_max),int(y_max)
            
            # pt1 = [x_min, y_min]
            # pt2 = [x_max,y_max]
            # # Fill the polygon on the dynamic_mask
            # cv2.rectangle(dynamic_mask, pt1,pt2, color=(255))  # Fill with 
            # cv2.imwrite('./tmp.png',dynamic_mask)
            # Write the coordinates to the file in xyxy format
            if x_min < 0 or y_min <0 or x_max <0 or y_max < 0:
                continue
            if x_max > width or y_max > height:
                continue 
            
            f.write(f"{x_min} {y_min} {x_max} {y_max}\n")
    return 
    # return dynamic_mask

def view_points_depth(points, view, normalize):
    #
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    viewpad = np.eye(4)
    viewpad[: view.shape[0], : view.shape[1]] = view
    nbr_points = points.shape[1]
    points = np.dot(viewpad, points)
    points = points[:3, :]
    depth = points[2, :]
    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)
    return points, depth

if __name__=='__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(description="Prepare novel trajectories gt labels.")
        parser.add_argument('--exp_name', type=str, default='pvg_example', help='Experiment name')
        parser.add_argument('--exp_root', type=str, default='/mnt/data-2/users/zhaoguosheng/1-code/17-drivedreamer4d_release/exp', help='Path to experiment root') 
        parser.add_argument('--scene_ids', type=str, nargs='+', default=None)
        parser.add_argument('--data_root', type=str, default='/mnt/data-2/users/zhaoguosheng/1-code/17-drivedreamer4d_release/data/waymo/processed/validation', help='Data path')
        parser.add_argument('--save_root', type=str, default='/mnt/data-2/users/zhaoguosheng/1-code/17-drivedreamer4d_release/results', help='Results path')
        return parser.parse_args()

    args = parse_args()
    prepare_novel_gt_label(args)