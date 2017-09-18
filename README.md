# IPSRO : Integrated Perception for Service RObots
#### Ver 1.0 (2017.09.06) by Jinyoung Choi 

## Description

IPSRO is deep learning based integrated perception framework for social service robots

It contains state-of-the-art object detector, human pose estimator, human re-identification, object captioning modules.

It can not only detect dozens of everyday life objects but also provide useful tags such as pose, identity, gender, cloth color, specific instance of objects, ETC..

We won the 1st place in RoboCup2017@Home Social Standard Platform League using this framework.

## Citations

* Object detector : https://github.com/thtrieu/darkflow

* Pose estimator : https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation

* Person re-identification : https://github.com/Ning-Ding/Implementation-CVPR2015-CNN-for-ReID

* Captioning : https://github.com/jcjohnson/densecap

## Requirements

Server : ROS Indigo, Ubuntu 14.04, Python 2.7, GPU with 8Gb memory or higher

Robot : Kinect sensor (or Asus axtion)

## Installation

Install below from their websites.
* Tensorflow r1.1 or higher
* Darkflow (https://github.com/thtrieu/darkflow) (intall option 3)

Install below as follows.

* Torch 
from http://torch.ch/docs/getting-started.html (choose 'yes' when installer asks something about path)

* Others
```
sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose

pip install Cython

sudo apt-get install libhdf5-dev libblas-dev liblapack-dev gfortran

pip install h5py

pip install keras

luarocks install nn

luarocks install image

luarocks install lua-cjson

luarocks install https://raw.githubusercontent.com/qassemoquab/stnbhwd/master/stnbhwd-scm-1.rockspec

luarocks install https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/torch-rnn-scm-1.rockspec

luarocks install cutorch

luarocks install cunn

luarocks install cudnn

luarocks install md5

luarocks install --server=http://luarocks.org/dev torch-ros

pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp27-none-linux_x86_64.whl 

pip install torchvision

```

* Download pre-trained weight files
1. Object detector : 
    1) download ```https://pjreddie.com/media/files/yolo.weights``` 
    2) put it in ```src/object_model```
2. Pose estimation : 
    1) download ```http://posefs1.perception.cs.cmu.edu/Users/ZheCao/pose_iter_440000.caffemodel```
    2) put it in ```src/pose_model/model/_trained_COCO/```
    3) download ```http://posefs1.perception.cs.cmu.edu/Users/ZheCao/pose_iter_146000.caffemodel```
    4) put it in ```src/pose_model/model/_trained_MPI/```
    5) download ```https://www.dropbox.com/s/ae071mfm2qoyc8v/pose_model.pth?dl=0```
    6) put it in ```src/pose_model/```
3. Re-identification : no need to download (included)
4. captioning : 
    1) download ```http://cs.stanford.edu/people/jcjohns/densecap/densecap-pretrained-vgg16.t7.zip```
    2) unzip it
    3) put unzipped file in ```src/captioning_model/data/models/densecap/```

## Usage

 1. compile the catkin package.

 2. Modify src/DIP_config.txt

```
show_integrated_perception : True or False (if True, DIP will visualize its perception)
perception_topic : DIP/perception (topic for integrated perception output)
rgb_topic : pepper_robot/camera/front/image_raw  (topic for your sensor's RGB image)
depth_topic : pepper_robot/camera/depth/image_raw  (topic for your sensor's Depth image)
obj_topic : DIP/objects   (topic for object detection results)
use_loc : True   (if True, DIP will calculate the locations of objects wrt robot/map/odometry and tag them automatically)
show_od : False   (if True, DIP will visualize its object detection results)
reid_target_topic : DIP/reid_targets   (topic for reid targets. send object_array to assign targets)
reid_topic : DIP/people_identified   (topic for re-identification results)
reid_thr : 0.75   (threshold for re-identification)
pose_topic : DIP/people_w_pose   (topic for pose estimation results)
captioning_topic : DIP/objects_w_caption   (topic for object captioning results)
captioning_request_topic : DIP/captioning_request   (topic for scene description request)
captioning_response_topic : DIP/captioning_response   (topic for scene description responses)
captioning_keywords_topic : DIP/captioning_keywords   (topic for keywords that will be extracted from captions)
```
    
 3. roslaunch dip_jychoi DIP_jychoi.launch

## Object and Object_array messages

We use custom message for individual object and array of objects

see msgs/objs.msg , msgs/objs_array.msg

We also use string_array custom message to send keywords to captioning module.
