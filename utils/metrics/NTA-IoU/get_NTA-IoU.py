import os
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union
import argparse
from ntaiou_utils import detect_novel_label,prepare_novel_gt_label

def calculate_iou(boxA, boxB):   
    x_inter_min = max(boxA[0], boxB[0])  
    y_inter_min = max(boxA[1], boxB[1])  
    x_inter_max = min(boxA[2], boxB[2])  
    y_inter_max = min(boxA[3], boxB[3])  
    inter_area = max(0, x_inter_max - x_inter_min) * max(0, y_inter_max - y_inter_min)  
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])  
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])  
    union_area = boxA_area + boxB_area - inter_area  
    iou = inter_area / union_area  
    return iou

def calculate_scenario_averages(averages):
    acc_avg = np.mean([v for k, v in averages.items() if 'acc' in k])
    dec_avg = np.mean([v for k, v in averages.items() if 'dec' in k])
    change_avg = np.mean([v for k, v in averages.items() if 'change_lane' in k])
    return acc_avg, dec_avg, change_avg

def find_closest_box(gt_box, ge_boxes):
    gt_center = [(gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2]
    min_distance = 10
    closest_box = None
    for ge_box in ge_boxes:
        ge_center = [(ge_box[0] + ge_box[2]) / 2, (ge_box[1] + ge_box[3]) / 2]
        distance = np.linalg.norm(np.array(gt_center) - np.array(ge_center))
        if distance < min_distance:
            min_distance = distance
            closest_box = ge_box
    return closest_box if min_distance <= 10 else None

def main(args):
    exp_name = args.exp_name
    exp_root = args.exp_root
    scene_ids = args.scene_ids
    save_root = args.save_root
    
    exp_root = os.path.join(exp_root,exp_name)
    if scene_ids is None:
        scene_ids = os.listdir(exp_root)
        scene_ids.sort()

    detect_root = os.path.join(save_root,exp_name,'detect_label')
    gt_root = os.path.join(save_root,'novel_gt_label')
    save_dir = os.path.join(save_root,exp_name,'csv')
    os.makedirs(save_dir,exist_ok=True)

    detect_dirs = []
    gt_dirs = []
    for scene in scene_ids:
        detect_dir = os.path.join(detect_root,scene)
        detect_dirs.append(detect_dir)
        gt_dir = os.path.join(gt_root,scene)
        gt_dirs.append(gt_dir)
 
    results = {}
    # Process each specified scenario
    for i in range(len(detect_dirs)):
        detect_dir = detect_dirs[i]
        gt_dir = gt_dirs[i]
        novel_trajs = os.listdir(detect_dir)
        for novel_traj in novel_trajs:
            detect_dir = os.path.join(detect_dir,novel_traj)
            gt_dir = os.path.join(gt_dir,novel_traj)

            assert os.path.exists(gt_dir), print('Please check the novel gt labels')
            assert os.path.exists(detect_dir), print('Please check the detect labels')
            gt_files = os.listdir(gt_dir) 

            key_name = scene+','+novel_traj
            results[key_name] = []
            for file in gt_files:
                if file.endswith(".txt"):
                    gt_path = os.path.join(gt_dir, file)
                    detect_path = os.path.join(detect_dir, file)
                    gt_boxes, detect_boxes = [], []

                    with open(gt_path, 'r') as f:
                        for line in f:
                            x1, y1, x2, y2 = map(int, line.strip().split())
                            gt_boxes.append([x1, y1, x2, y2])

                    with open(detect_path, 'r') as f:
                        for line in f:
                            x1, y1, x2, y2 = map(int, line.strip().split())
                            detect_boxes.append([x1, y1, x2, y2])

                    for gt_box in gt_boxes:
                        closest_detect_box = find_closest_box(gt_box, detect_boxes)
                        if closest_detect_box:
                            iou = calculate_iou(gt_box, closest_detect_box)
                            results[key_name].append(iou)
                        else:
                            results[key_name].append(0)

                
    # Calculate average GIoU loss for each scenario
    res = []
    for k,v in results.items():
        scene,novel_traj = k.split(',')
        iou = np.average(v)
        res.append([scene,novel_traj,iou])
    # Save results to a CSV file
    df = pd.DataFrame(res, columns=['Scene','Novel Trajectory', 'NTA-IoU'])
    
    df.to_csv(os.path.join(save_dir,'NTA-IoU.csv'), index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate NTA-IoU.")
    
    parser.add_argument('--model_path', type=str, default='/PATH/TO/YOUR/YOLO11/MODEL', help='Path to the YOLO11 model file')
    parser.add_argument('--exp_name', type=str, default='pvg_example', help='Experiment name')
    parser.add_argument('--exp_root', type=str, default='./exp', help='Path to experiment root') 
    parser.add_argument('--scene_ids', type=str, nargs='+', default=None)
    parser.add_argument('--data_root', type=str, default='./data/waymo/processed/validation', help='Data path')
    parser.add_argument('--save_root', type=str, default='./results', help='Results path')
    return parser.parse_args()

if __name__=='__main__':
    args=parse_args()
    prepare_novel_gt_label(args)
    detect_novel_label(args)
    main(args)




