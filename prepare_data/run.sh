#!/bin/bash


#PNet
#cd /data/ai/FasterRCNN/MTCNN-Tensorflow-master/prepare_data
python gen_12net_data.py
python gen_landmark_aug_12.py
python gen_imglist_pnet.py
python gen_PNet_tfrecords.py
cd /data/ai/FasterRCNN/MTCNN-Tensorflow-master/train_models
python train_PNet.py
#RNet
cd /data/ai/FasterRCNN/MTCNN-Tensorflow-master/prepare_data
python gen_hard_example_RNet.py
python gen_landmark_aug_24.py
python gen_imglist_pnet.py
python gen_RNet_tfrecords.py
python gen_RNet_tfrecords_pos.py
python gen_RNet_tfrecords_part.py
python gen_RNet_tfrecords_neg.py
cd /data/ai/FasterRCNN/MTCNN-Tensorflow-master/train_models
python train_RNet.py

#ONet:
cd /data/ai/FasterRCNN/MTCNN-Tensorflow-master/prepare_data
python gen_hard_example_ONet.py
python gen_landmark_aug_48.py
python gen_imglist_onet.py
python gen_ONet_tfrecords.py
python gen_ONet_tfrecords_pos.py 
python gen_ONet_tfrecords_part.py 
python gen_ONet_tfrecords_neg.py
cd /data/ai/FasterRCNN/MTCNN-Tensorflow-master/train_models
python train_ONet.py
