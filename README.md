# YOLOv3 on Deep Fashion 2 data set 

This project is heavily motivated form Erik Linder-Norén's YoloV3 implementation in pytorch
https://github.com/eriklindernoren/PyTorch-YOLOv3
This is a python  Implementation   of DarkNet
https://github.com/AlexeyAB/darknet

## Installation
##### Clone and install requirements
    $ copy the files
    $ sudo pip3 install -r requirements.txt



##### Download Deep Fashion 2 data set 
    $ https://github.com/switchablenorms/DeepFashion2
    $ Once you are registering in the portal you will receive a password for unzipping the files 
    
## Convert Deep fashion annotations to YoloV3 format

Deep fashion annotation is json file in the following format 
{"item2": {"segmentation": [[460, 438, 374, 484, 251, 520, 269, 586, 298, 622, 410, 623, 410, 567, 413, 591, 420, 623, 465, 622, 456, 561, 466, 504, 460, 438], [374, 484, 251, 520, 269, 586, 298, 622, 410, 623, 410, 567, 374, 484], [460, 438, 374, 484, 410, 567, 413, 591, 420, 623, 465, 622, 456, 561, 466, 504, 460, 438]], "scale": 2, "viewpoint": 2, "zoom_in": 3, "landmarks": [251, 520, 1, 374, 484, 1, 460, 438, 1, 269, 586, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 410, 567, 1, 413, 591, 2, 0, 0, 0, 0, 0, 0, 456, 561, 2, 0, 0, 0], "style": 0, "bounding_box": [249, 423, 466, 623], "category_id": 8, "occlusion": 2, "category_name": "trousers"}, "source": "user", "pair_id": 1, "item1": {"segmentation": [[257, 35, 261, 89, 228, 123, 137, 103, 45, 91, 1, 176, 0, 332, 47, 447, 151, 401, 141, 366, 129, 328, 141, 364, 219, 485, 274, 603, 401, 590, 467, 502, 442, 336, 369, 195, 348, 138, 363, 163, 372, 197, 433, 137, 396, 92, 341, 35, 257, 35], [1, 176, 0, 332, 47, 447, 151, 401, 141, 366, 129, 328, 1, 176], [348, 138, 363, 163, 372, 197, 433, 137, 396, 92, 341, 35, 348, 138]], "scale": 3, "viewpoint": 2, "zoom_in": 2, "landmarks": [182, 54, 1, 45, 91, 1, 137, 103, 1, 228, 123, 1, 261, 89, 1, 257, 35, 1, 0, 0, 0, 0, 0, 0, 47, 447, 2, 151, 401, 2, 141, 366, 2, 129, 328, 2, 141, 364, 2, 219, 485, 2, 274, 603, 2, 401, 590, 2, 0, 0, 0, 442, 336, 2, 369, 195, 1, 348, 138, 1, 363, 163, 1, 372, 197, 1, 433, 137, 2, 396, 92, 2, 341, 35, 1], "style": 1, "bounding_box": [0, 29, 466, 622], "category_id": 1, "occlusion": 2, "category_name": "short sleeve top"}}

We need to convert the same to YoloV3 annotation txt format 
0 0.49786324786324787 0.5216346153846154 0.9957264957264957 0.9503205128205128
7 0.7638888888888888 0.8381410256410257 0.4636752136752137 0.32051282051282054

Each row in the annotation file should define one bounding box, using the syntax label_idx x_center y_center width height. The coordinates should be scaled [0, 1], and the label_idx should be zero-indexed and correspond to the row number of the class name in data/custom/classes.names

I have created a python script for converting Deep fashion annotation to YoloV3

	$ cd scripts/
	$ python3 df2yolo_annotation.py --image_folder folder_containing_images_and_annos_in_DF2_format

After the previous steps you can see 3 folder in --image_folder
a) images b) annos c) labels


	$ copy images and labels folder into /data/custom folder 
	$ edit data/custom/train.txt & data/custom/valid.txt with the corresponding train and validation images.
	$ You can make use of the script scripts/generate_test_validation_list.py for generating train.txt and valid.txt



## Train YoloV3 in custom data 

1. For training cfg/yolov4-custom.cfg download the pre-trained weights-file (162 MB): yolov4.conv.137 (Google drive mirror yolov4.conv.137 )

2. Create file config/yolo-custom.cfg with the same content as in config/yolo-custom.cfg (or copy config/yolo-custom.cfg to config/yolo-custom.cfg)

3. change line batch to batch=64
4. change line subdivisions to subdivisions=16
5. change line max_batches to 26000(classes*2000 , Deep fashion there are 13 classes 
6. change line steps to 80% and 90% of max_batches 
7. set network size width=416 height=416 or any value multiple of 32: https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L8-L9
8. change line classes=13 in each of  [yolo]-layers:
9. change [filters=255] to filters=(classes + 5)x3  == 54 in the 3 [convolutional] before each [yolo] layer
10. when using [Gaussian_yolo] layers, change [filters=57] filters=(classes + 9)x3 ==66 in the 3 [convolutional] before each [Gaussian_yolo] layer

For more details please follown darknet page
https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects



## Create df2.names with the list of classes
gedit /data/df2.names 

short sleeve top
long sleeve top
short sleeve outwear
long sleeve outwear
vest
sling
shorts
trousers
skirt
short sleeve dress
long sleeve dress
vest dress
sling dress

## create custom.data files
gedit /config/custom.data

classes= 13
train=data/custom/train.txt
valid=data/custom/valid.txt
names=data/custom/classes.names


## start training 
Once all the above steps are completed start training
	$ python3 train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --pretrained_weights weights/darknet53.conv.74

	$ After each --checkpoint_interval mentioned in the train.py a weights file will be saved in the checkpoints folder.


#### Training log
```
---- [Epoch 7/100, Batch 7300/14658] ----
+------------+--------------+--------------+--------------+
| Metrics    | YOLO Layer 0 | YOLO Layer 1 | YOLO Layer 2 |
+------------+--------------+--------------+--------------+
| grid_size  | 16           | 32           | 64           |
| loss       | 1.554926     | 1.446884     | 1.427585     |
| x          | 0.028157     | 0.044483     | 0.051159     |
| y          | 0.040524     | 0.035687     | 0.046307     |
| w          | 0.078980     | 0.066310     | 0.027984     |
| h          | 0.133414     | 0.094540     | 0.037121     |
| conf       | 1.234448     | 1.165665     | 1.223495     |
| cls        | 0.039402     | 0.040198     | 0.041520     |
| cls_acc    | 44.44%       | 43.59%       | 32.50%       |
| recall50   | 0.361111     | 0.384615     | 0.300000     |
| recall75   | 0.222222     | 0.282051     | 0.300000     |
| precision  | 0.520000     | 0.300000     | 0.070175     |
| conf_obj   | 0.599058     | 0.622685     | 0.651472     |
| conf_noobj | 0.003778     | 0.004039     | 0.004044     |
+------------+--------------+--------------+--------------+
Total Loss 4.429395
---- ETA 0:35:48.821929
```

#### Tensorboard
Track training progress in Tensorboard:
* Initialize training
* Run the command below
* Go to http://localhost:6006/

```
$ tensorboard --logdir='logs' --port=6006
```



## Testing the algorithm 
Place your test images in  data/samples folder
	$ python3 detect.py --image_folder data/samples/ --model_def config/yolov3-custom.cfg --weights_path weights/yolov3-df2_last.weights --class_path data/df2.names
	$ You need to specify the best weights after full epochs. 



## Credit

### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Project Webpage]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation]](https://github.com/pjreddie/darknet)

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
