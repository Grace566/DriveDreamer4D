from ultralytics import YOLO
import cv2
import os
import imageio

def detect_novel_label(args):
    model_path =args.model_path
    exp_name = args.exp_name
    exp_root = args.exp_root
    scene_ids = args.scene_ids
    save_root = args.save_root
    # Load a pretrained YOLO model
    model = YOLO(model_path)

    exp_root = os.path.join(exp_root,exp_name)
    if scene_ids is None:
        scene_ids = os.listdir(exp_root)
        scene_ids.sort()
    
    save_root = os.path.join(save_root,exp_name,'detect_label')
    os.makedirs(save_root, exist_ok=True)  # Ensure the root output directory exists

    # Define the target size for resizing
    target_size = (480, 320)  # width, height

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
        for novel_traj in novel_trajs:
            novel_img_dir = os.path.join(img_dir,novel_traj)
            fns = os.listdir(novel_img_dir)
            
            for img_name in fns:
                img_path = os.path.join(novel_img_dir,img_name)
                image = cv2.imread(img_path)
                # Resize the image to 480x320
                resized_image = cv2.resize(image, target_size)
            
                # Prepare the output directory structure similar to the input structure
                save_dir = os.path.join(save_root, scene,novel_traj)
                os.makedirs(save_dir, exist_ok=True)
                txt_name = img_name.replace('.png','.txt')  # Extract the image name without extension
                output_file = os.path.join(save_dir, txt_name)
                if os.path.exists(output_file):
                    continue  
                results = model(resized_image)
                with open(output_file, 'w') as f:
                # View results and save boxes with confidence > 0.5
                    for r in results:
                        for box in r.boxes:
                            conf = box.conf  # Get the confidence score
                            if conf > 0.5:  # Only save boxes with confidence > 0.5
                                # Extract xyxy coordinates (top-left and bottom-right corner)
                                xyxy = box.xyxy[0].cpu().numpy()  # Get the bounding box coordinates
                                x1, y1, x2, y2 = map(int, xyxy)  # Convert to integers
                                
                                # Write the coordinates to the file in the format: x1, y1, x2, y2
                                f.write(f"{x1} {y1} {x2} {y2}\n")
    print("Processing completed for all images.")

if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(description="Prepare novel trajectories gt labels.")
        parser.add_argument('--model_path', type=str, default='/mnt/data-2/users/zhaoguosheng/1-code/1-drivestudio/yolo11x.pt', help='Path to the TwinLiteNet model file')
        parser.add_argument('--exp_name', type=str, default='pvg_example', help='Experiment name')
        parser.add_argument('--exp_root', type=str, default='/mnt/data-2/users/zhaoguosheng/1-code/17-drivedreamer4d_release/exp', help='Path to experiment root') 
        parser.add_argument('--scene_ids', type=str, nargs='+', default=None)
        parser.add_argument('--save_root', type=str, default='/mnt/data-2/users/zhaoguosheng/1-code/17-drivedreamer4d_release/results', help='Results path')
        return parser.parse_args()

    args = parse_args()
    detect_novel_label(args)
