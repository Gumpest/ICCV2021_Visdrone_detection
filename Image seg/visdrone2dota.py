import os
import cv2
from tqdm import tqdm
import json

categories = {'0':"pedestrian", '1': "people", '2': "bicycle",
              '3': "car", '4': "van", '5': "truck",
              '6': "tricycle", '7': "awning-tricycle",
              '8': "bus", '9': "motor"}

def convert_visdronelabel_to_dota(dir, output_dir):
    train_dir = os.path.join(dir, "VisDrone2019-DET-train")
    val_dir = os.path.join(dir, "VisDrone2019-DET-val")
    # test_dir = os.path.join(dir, "VisDrone2019-DET-test-dev")
    test_dir = '/home/zy/runs/detect/exp2_608/labels/'
    test_annotations = test_dir
    train_annotations = os.path.join(train_dir, "annotations")
    val_annotations = os.path.join(val_dir, "annotations")
    # test_annotations = os.path.join(test_dir, "annotations")
    train_images = os.path.join(train_dir, "images")
    val_images = os.path.join(val_dir, "images")
    test_images = os.path.join(test_dir, "images")

    for mode in ["test"]:
        print(f"start loading {mode} data...")
        if mode == "train":
            set = os.listdir(train_annotations)
            annotations_path = train_annotations
            images_path = train_images
        elif mode == "test":
            set = os.listdir(test_annotations)
            annotations_path = test_annotations
            images_path = test_images
        else:
            set = os.listdir(val_annotations)
            annotations_path = val_annotations
            images_path = val_images
        for i in tqdm(set):
            f = open(annotations_path + "/" + i, "r")
            with open(os.path.join(output_dir,i), 'a') as ff:
                # ff.write('imagesource:GoogleEarth'+'\n')
                # ff.write('gsd:null'+'\n')
                for line in f.readlines():
                    line = line.replace("\n", "")
                    if line.endswith(","):  # filter data
                        line = line.rstrip(",")
                    line_list = [i for i in line.split(",")]
                    if line_list[5]=='0' or line_list[5]=='11':
                        continue
                    ff.write(line_list[0]+' '+line_list[1]+' ')
                    ff.write(str(int(line_list[0])+int(line_list[2]))+' '+line_list[1]+' ')
                    ff.write(str(int(line_list[0])+int(line_list[2]))+' '+str(int(line_list[1])+int(line_list[3]))+' ')
                    ff.write(line_list[0]+' '+str(int(line_list[1])+int(line_list[3]))+' ')
                    ff.write(categories[str(int(line_list[5])-1)]+' 0'+'\n')
            ff.close()
            f.close()

# convert_visdronelabel_to_dota('/home/lxc/visdrone', '/home/lxc/visdrone/VisDrone2019-DET-val/ann_dota')
# convert_visdronelabel_to_dota('/home/lxc/visdrone', '/home/lxc/visdrone/VisDrone2019-DET-train/ann_dota')
convert_visdronelabel_to_dota('/home/lxc/visdrone', '/home/lxc/visdrone/test_merge/dota_ann')
