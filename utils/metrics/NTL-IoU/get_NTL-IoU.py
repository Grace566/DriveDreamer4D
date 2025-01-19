import torch
import numpy as np
from tqdm.autonotebook import tqdm
import os
import os
import torch
from model import TwinLite as net
import cv2
import mmengine
import imageio
from IOUEval import SegmentationMetric
import pandas as pd
import argparse

MAX_MAP = 75

line_cls_map = {
   'road_edge':0,
   'road_line':2,
   'crosswalk':1
}
# global LL
LL=SegmentationMetric(2)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def view_points_depth(points, view, normalize):
    # view = view.to('cpu')
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

def get_hdmap(vectors,c2w,intrinsic,height=1280,width=1920):
    try:
        world2cam = np.linalg.inv(c2w.cpu())
    except:
        world2cam = np.linalg.inv(c2w)
    map_canvas = np.zeros((height,width),dtype=np.uint8)
    for pts_w, _type in vectors:
        pts = world2cam@pts_w.T
        
        # filter points out of map
        keep_index = []
        dist = np.sqrt(np.square(pts[:-1]).sum(0))
        for i in range(pts.shape[1]):
            if dist[i]<MAX_MAP and abs(pts[1,i])<10:
                keep_index.append(i)
        if len(keep_index)<2:
            continue
        index = np.array(keep_index)
        flag = (index[1:]-index[:-1])>1
        split_idx = [0]
        if flag.any():
            split_idx += list(np.argwhere(flag).reshape(-1)+1)
            pass
        
        split_idx+=[len(keep_index)]
        for idx_start,idx_end in zip(split_idx[:-1],split_idx[1:]):
            idx = keep_index[idx_start:idx_end]
            if len(idx)<2:
                continue
            _pts = pts[:,idx]
            # _pts_2 = ego2cam@ego_pts[:,idx]
            _pts,_depth = view_points_depth(_pts,intrinsic,normalize=True)
            _pts = _pts[:, _depth > 1e-3]
            _pts = _pts[:2, :]
            _pts=_pts.T
            cv2.polylines(map_canvas, [_pts.astype(np.int32)], False, color=255, thickness=int(20*height/1280))
    return map_canvas

def Run(model,img,label,ll_acc_seg,ll_IoU_seg,ll_mIoU_seg):
    img = cv2.resize(img, (640, 360))
    img_rs=img.copy()
    
    label = cv2.resize(label, (640, 360))
    _,seg_b = cv2.threshold(label,1,255,cv2.THRESH_BINARY_INV)
    _,seg = cv2.threshold(label,1,255,cv2.THRESH_BINARY)
    seg_b = torch.from_numpy(seg_b)[None].float()/255.0
    seg = torch.from_numpy(seg)[None].float()/255.0
    seg_ll = torch.stack((seg_b[0], seg[0]),0)[None]
    _,ll_gt=torch.max(seg_ll, 1)

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img=torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)  # add a batch dimension
    img=img.cuda().float() / 255.0
    img = img.cuda()
    with torch.no_grad():
        img_out = model(img)
    x0=img_out[0]
    x1=img_out[1]

    _,da_predict=torch.max(x0, 1)
    _,ll_predict=torch.max(x1, 1)

    LL.reset()
    LL.addBatch(ll_predict.cpu(), ll_gt)

    ll_acc = LL.pixelAccuracy()
    ll_IoU = LL.IntersectionOverUnion()
    ll_mIoU = LL.meanIntersectionOverUnion()
    # DA = da_predict.byte().cpu().data.numpy()[0]*255
    ll = ll_predict.byte().cpu().data.numpy()[0]*255
    # img_rs[DA>100]=[255,0,0]
    img_rs[ll>100]=[0,255,0]

    ll_acc_seg.update(ll_acc,1)
    ll_IoU_seg.update(ll_IoU,1)
    ll_mIoU_seg.update(ll_mIoU,1)
    
    return img_rs

def main(args):
    model_path = args.model_path
    exp_name = args.exp_name
    exp_root = args.exp_root
    data_root = args.data_root
    scene_ids = args.scene_ids
    save_root = args.save_root

    start_idx_map = {
        '005':120
    }
    model = net.TwinLiteNet()
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    exp_root = os.path.join(exp_root,exp_name)
    if scene_ids is None:
        scene_ids = os.listdir(exp_root)
        scene_ids.sort()
    
    results = []
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
            novel_img_dir = os.path.join(img_dir,novel_traj)
            fns = os.listdir(novel_img_dir)
            fns.sort()
            data_path = os.path.join(data_root,scene,'label.pkl')
            data_dict = mmengine.load(data_path)
            hdamps = data_dict['hdmap']
            calibs = data_dict['calibs']
            ll_acc_mix = AverageMeter()
            ll_IoU_mix = AverageMeter()
            ll_mIoU_mix = AverageMeter()
            for j,img_name in tqdm(enumerate(fns)):
                i = start_idx+j
                vectors = hdamps[i]
                calib =calibs[i]
                cur_ego2world = calib['ego2world']
                cam2cur_ego = calib['cam2ego']
                intrinsic = calib['intrinsic']
                if j==0:
                    eog_02world = cur_ego2world
                    world2ego_0 = np.linalg.inv(eog_02world).astype(np.float128)
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
                label = get_hdmap(vectors,ref_c2w.astype(np.float64),intrinsic)
                img = cv2.imread(os.path.join(novel_img_dir,img_name))
                Run(model,img,label,ll_acc_mix,ll_IoU_mix,ll_mIoU_mix)
            results.append((scene,novel_traj,ll_mIoU_mix.avg))
            
    df = pd.DataFrame(results,columns=['Scenes','Novel Trajectory','NTL-IoU'])
    save_path = os.path.join(save_root,exp_name,'csv')
    os.makedirs(save_path,exist_ok=True)
    df.to_csv(os.path.join(save_path,'NTL-IoU.csv'), index=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Detect lanes with TwinLiteNet model.")
    
    parser.add_argument('--model_path', type=str, default='/PATH/TO/YOUR/TWINLITENET/MODEL', help='Path to the TwinLiteNet model file')
    parser.add_argument('--exp_name', type=str, default='pvg_example', help='Experiment name')
    parser.add_argument('--exp_root', type=str, default='./exp', help='Path to experiment root') 
    parser.add_argument('--scene_ids', type=str, nargs='+', default=None)
    parser.add_argument('--data_root', type=str, default='./data/waymo/processed/validation', help='Data path')
    parser.add_argument('--save_root', type=str, default='./results', help='Results path')
    return parser.parse_args()


if __name__=='__main__':
    args=parse_args()
    main(args)