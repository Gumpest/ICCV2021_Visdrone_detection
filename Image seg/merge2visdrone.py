import os
import cv2
from tqdm import tqdm
import json

categories = {"pedestrian":'1', "people":'2', "bicycle":'3',
              "car":'4', "van":'5', "truck":'6',
              "tricycle":'7', "awning-tricycle":'8',
              "bus":'9', "motor":'10'}

def convert_mergelabel_to_visdrone(mergedir, output_dir):
    for cls in os.listdir(mergedir):
        f = open(os.path.join(mergedir,cls),'r')
        for line in f.readlines():
            line = line.replace("\n", "")
            line_list = [i for i in line.split(" ")]
            with open(os.path.join(output_dir,line_list[0]+'.txt'), 'a') as ff:
                ff.write(str(round(float(line_list[2])))+',')
                ff.write(str(round(float(line_list[3])))+',')
                ff.write(str(round(float(line_list[4])-float(line_list[2])))+',')
                ff.write(str(round(float(line_list[5])-float(line_list[3])))+',')
                ff.write(line_list[1]+',')
                ff.write(categories[cls.split('.')[0]]+',')
                ff.write('-1'+',')
                ff.write('-1'+'\n')

# convert_visdronelabel_to_dota('/home/lxc/visdrone', '/home/lxc/visdrone/VisDrone2019-DET-val/ann_dota')
# convert_visdronelabel_to_dota('/home/lxc/visdrone', '/home/lxc/visdrone/VisDrone2019-DET-train/ann_dota')
convert_mergelabel_to_visdrone('/home/lxc/visdrone/test_merge/dota_merge/yolov5_nms', '/home/lxc/visdrone/test_merge/visdrone/yolov5')
