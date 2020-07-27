import numpy as np
from PIL import Image
import os
import json 
import glob
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, help="Folder containing image files")
    args = parser.parse_args()
    directory = args.image_folder + '/labels/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    annos_list = glob.glob(args.image_folder + '/annos/*.json')

    for anno in annos_list:
        len_split = len(anno.split('/'))
        file_name_jpg = anno.split('/')[len_split-1].split('.')[0]
        file_nanme_jpg_full = anno.split('annos')[0] + 'image/' + file_name_jpg + '.jpg'
        file_name_label_txt = directory + file_name_jpg + '.txt'
        img = Image.open(file_nanme_jpg_full)
        img_width = img.size[0]
        img_height = img.size[1]
        # print (len_split, img.size, file_nanme_jpg_full, file_name_label_txt)
        file = open(file_name_label_txt, "a")
        with open(anno) as json_file:
            data = json.load(json_file)
            for i in range(10):
                key = 'item{}'.format(i + 1)
                if key in data:
                    item_dic = data[key]
                    cat_id = item_dic['category_id']
                    bbox = item_dic['bounding_box']
                    string1 = str(cat_id-1)
                    string2 = str(bbox[0]/img_width)
                    string3 = str(bbox[1]/img_height)
                    string4 = str((bbox[2]-bbox[0])/img_width)
                    string5 = str((bbox[3]-bbox[1])/img_height)
                    file.write(string1 + ' ' + string2 + ' ' + string3 + ' ' + string4 + ' ' + string5 + '\n')
                    # print(cat_id, bbox)

        file.close()