## Description
This work is used for reproduce MTCNN,a Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks.

## Prerequisites
1. You need CUDA-compatible GPUs to train the model.
2. You should first download [WIDER Face](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) and [Celeba](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).**WIDER Face** for face detection and **Celeba** for landmark detection(This is required by original paper.But I found some labels were wrong in Celeba. So I use [this dataset](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm) for landmark detection).

## Dependencies
* Tensorflow 1.2.1
* TF-Slim
* Python 2.7
* Ubuntu 16.04
* Cuda 8.0

## Prepare For Training Data
1. Download Wider Face Training part only from Official Website , unzip to replace `WIDER_train` and put it into `prepare_data` folder.
2. Download landmark training data from [here]((http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm)),unzip and put them into `prepare_data` folder.
3. Run `prepare_data/gen_12net_data.py` to generate training data(Face Detection Part) for **PNet**.
4. Run `gen_landmark_aug_12.py` to generate training data(Face Landmark Detection Part) for **PNet**.
5. Run `gen_imglist_pnet.py` to merge two parts of training data.
6. Run `gen_PNet_tfrecords.py` to generate tfrecord for **PNet**.
7. After training **PNet**, run `gen_hard_example` to generate training data(Face Detection Part) for **RNet**.
8. Run `gen_landmark_aug_24.py` to generate training data(Face Landmark Detection Part) for **RNet**.
9. Run `gen_imglist_rnet.py` to merge two parts of training data.
10. Run `gen_RNet_tfrecords.py` to generate tfrecords for **RNet**.(**you should run this script four times to generate tfrecords of neg,pos,part and landmark respectively**)
11. After training **RNet**, run `gen_hard_example` to generate training data(Face Detection Part) for **ONet**.
12. Run `gen_landmark_aug_48.py` to generate training data(Face Landmark Detection Part) for **ONet**.
13. Run `gen_imglist_onet.py` to merge two parts of training data.
14. Run `gen_ONet_tfrecords.py` to generate tfrecords for **ONet**.(**you should run this script four times to generate tfrecords of neg,pos,part and landmark respectively**)

## Some Details
* When training **PNet**,I merge four parts of data(pos,part,landmark,neg) into one tfrecord,since their total number radio is almost 1:1:1:3.But when training **RNet** and **ONet**,I generate four tfrecords,since their total number is not balanced.During training,I read 64 samples from pos,part and landmark tfrecord and read 192 samples from neg tfrecord to construct mini-batch.
* It's important for **PNet** and **RNet** to keep high recall radio.When using well-trained **PNet** to generate training data for **RNet**,I can get 14w+ pos samples.When using well-trained **RNet** to generate training data for **ONet**,I can get 19w+ pos samples.
* Since **MTCNN** is a Multi-task Network,we should pay attention to the format of training data.The format is:
 
  [path to image][cls_label][bbox_label][landmark_label]
  
  For pos sample,cls_label=1,bbox_label(calculate),landmark_label=[0,0,0,0,0,0,....](total 37*2 zeros).

  For part sample,cls_label=-1,bbox_label(calculate),landmark_label=[0,0,0,0,0,.....].
  
  For landmark sample,cls_label=-2,bbox_label=[0,0,0,0],landmark_label(calculate).  
  
  For neg sample,cls_label=0,bbox_label=[0,0,0,0],landmark_label=[0,0,0,0,0,....](total 37*2 zeros).  

* Since the training data for landmark is less.I use transform,random rotate and random flip to conduct data augment(the result of landmark detection is not that good).

## Result

![result1.png](https://i.loli.net/2017/08/30/59a6b65b3f5e1.png)

![result2.png](https://i.loli.net/2017/08/30/59a6b6b4efcb1.png)

![result3.png](https://i.loli.net/2017/08/30/59a6b6f7c144d.png)

![reult4.png](https://i.loli.net/2017/08/30/59a6b72b38b09.png)

![result5.png](https://i.loli.net/2017/08/30/59a6b76445344.png)

![result6.png](https://i.loli.net/2017/08/30/59a6b79d5b9c7.png)

![result7.png](https://i.loli.net/2017/08/30/59a6b7d82b97c.png)

![result8.png](https://i.loli.net/2017/08/30/59a6b7ffad3e2.png)

![result9.png](https://i.loli.net/2017/08/30/59a6b843db715.png)

**Result on FDDB**
![result10.png](https://i.loli.net/2017/08/30/59a6b875f1792.png)

## License
MIT LICENSE

## References
1. Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, Yu Qiao , " Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks," IEEE Signal Processing Letter
2. [MTCNN-MXNET](https://github.com/Seanlinx/mtcnn)
3. [MTCNN-CAFFE](https://github.com/CongWeilin/mtcnn-caffe)
4. [deep-landmark](https://github.com/luoyetx/deep-landmark)

关于更改关键点个数说明

一、	MTCNN训练样本的关键在于改变数据准备及自定义网络的关键点个数.
1.	改变自定义网络，仅需在train_models/mtcnn_model.py中将PNET,RNET,ONET的landmark_pred = slim.conv2d(net,num_outputs=74)改为74（37*2）即可。Train.py中 landmark_target=74,生成的74个关键点数据集txt是wider_face_train.txt
2.	改变数据准备之Pnet
a)	prepare_data/gen_12net_data.py生成bounding_box数据，用wider_train，直接生成即可。
b)	run gen_landmark_aug_12.py生成关键点数据，需要用到自定义数据，且要生成格式为trainImageList.txt,testImageList.txt来分别存储训练及测试数据。其中，每一行的格式为“图片名称(1) boungding box(4个坐标点) landmark各点坐标（74个）。且txt对应的图片放入prepare_data下。对于涉及的BBox_utils.py需要修改getDataFromTxt ()中landmark = np.zeros((37, 2))及  for index in range(0, 37)，特别注意bbox = (components[1], components[2], components[3], components[4])句，需要与自定义数据集对应上。而Landmark_utils.py，需要修改randomShift()中diff = np.random.rand(37, 2)及randomShiftWithArgument()中landmarkPs = np.zeros((N, 37, 2))。
c)	run gen_imglist_pnet.py需要修改的是with open(os.path.join(dir_path, "%s" %(net),"train_%s_landmark.txt" % (net)), "a") as f: 原本的“w”会使得数据读取不完整。
d)	run gen_PNet_tfrecords.py需要修改的是get_dataset()中初始化bbox[‘x1’]=0 bbox[‘y1’]=0…….bbox[‘x37’]=0,bbox[‘y37’]=0,
if len(info)==76:     bbox['x1'] =f loat(info[2])…..
bbox[‘y37’]=float(info[75])
read_tfrecord_v2.py中 read_single_tfrecord()----image/landmark': tf.FixedLenFeature([74],tf.float32) 
landmark = tf.reshape(landmark,[batch_size,74])
tfrecord_utils.py中修改_convert_to_example_simple ()
landmark=[bbox['x1'],bbox['y1']….. bbox['x37'], bbox['y37']]
3 . 改变数据准备之Rnet
a)	run gen_hard_example_RNet.py 需要修改parse_args() parser.add_argument ('--test_mode'为’PNnet’, if __name__ == '__main__':中net = 'RNet'。否则加载Pnet的模型会出现fcl/alpha is not defined或者权重与网络结构不匹配。
b)	run gen_landmark_aug_24.py 及run gen_imglist_pnet.py同Pnet
c)	run gen_RNet_tfrecords.py 初始利用的landmark是24/landmark_24_aug.txt里，会出现segmentation fault，修改get_dataset(dir, net='RNet')，item = 'imglists/RNet/train_%s_landmark.txt' % net，同时，if __name__ == '__main__': net='RNet'
d)	run train_Rnet.py时找不到pos/neg/part_landmark.tfrecord,需要手动将data_dir改为train_RNet_landmark.tfrecord_shuffle.另一种解决办法是，生成gen_RNet_tfrecords_pos/neg/part.py 将24/pos_24.txt 转换为pos_landmark.tfrecord_shuffle
3.	改变数据之ONet
a)	run gen_hard_example_ONet.py 需要修改parse_args() parser.add_argument ('--test_mode'为’RNnet’, if __name__ == '__main__':中net = 'ONet'。否则加载Rnet的模型会出现fcl/alpha is not defined或者权重与网络结构不匹配。
parser.add_argument('--epoch', default=[18, 22, 22]以获取最终迭代的权重。
b)	run gen_landmark_aug_24.py 及run gen_imglist_pnet.py同Pnet

