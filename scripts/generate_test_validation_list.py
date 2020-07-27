import numpy as np
from PIL import Image
import os
import json 
import glob
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, help="Folder containing image files")
    parser.add_argument("--ratio", type=str, help="Train to validation ratio")
    args = parser.parse_args()
    annos_list = glob.glob(args.image_folder + '/labels/*.txt')
    file_training = open('/home/tosebas/Cloth_Detection_YoloV3/data/custom/train.txt', 'a')
    file_validation = open('/home/tosebas/Cloth_Detection_YoloV3/data/custom/valid.txt','a')
    ratio = int(args.ratio)

    for anno in annos_list:

        len_split = len(anno.split('/'))
        file_name = anno.split('/')[len_split-1].split('.')[0]
        print(anno, file_name)

        if(int(file_name)% ratio !=0):
            file_training.write('data/custom/images/' + file_name +'.jpg'+'\n')
        else:
            file_validation.write('data/custom/images/' + file_name + '.jpg' + '\n')
    file_training.close()
    file_validation.close()


