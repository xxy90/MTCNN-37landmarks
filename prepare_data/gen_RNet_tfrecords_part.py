#coding:utf-8
import os
import random
import sys
import time

import tensorflow as tf

from tfrecord_utils import _process_image_withoutcoder, _convert_to_example_simple


def _add_to_tfrecord(filename, image_example, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    print('---', filename)
    #imaga_data:array to string
    #height:original image's height
    #width:original image's width
    #image_example dict contains image's info
    image_data, height, width = _process_image_withoutcoder(filename)
    example = _convert_to_example_simple(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, net):
    #st = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #return '%s/%s_%s_%s.tfrecord' % (output_dir, name, net, st)
    #return '%s/train_PNet_landmark.tfrecord' % (output_dir)
    return '%s/part_landmark.tfrecord' % (output_dir)
    

def run(dataset_dir, net, output_dir, name='MTCNN', shuffling=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    
    #tfrecord name 
    tf_filename = _get_output_filename(output_dir, name, net)
    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    # GET Dataset, and shuffling.
    dataset = get_dataset(dataset_dir, net=net)
    # filenames = dataset['filename']
    if shuffling:
        tf_filename = tf_filename + '_shuffle'
        #andom.seed(12345454)
        random.shuffle(dataset)
    # Process dataset files.
    # write the data to tfrecord
    print 'lala'
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, image_example in enumerate(dataset):
            sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(dataset)))
            sys.stdout.flush()
            filename = image_example['filename']
            _add_to_tfrecord(filename, image_example, tfrecord_writer)
    tfrecord_writer.close()
    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the MTCNN dataset!')


def get_dataset(dir, net='24'):
    #item = 'imglists/PNet/train_%s_raw.txt' % net
    #item = 'imglists/RNet/train_%s_landmark.txt' % net
    item = '%s/part_%s.txt' % (net,net)
    print item 
    dataset_dir = os.path.join(dir, item)
    imagelist = open(dataset_dir, 'r')

    dataset = []
    for line in imagelist.readlines():
        info = line.strip().split(' ')
        data_example = dict()
        bbox = dict()
        data_example['filename'] = info[0]
        data_example['label'] = int(info[1])
        bbox['xmin'] = 0
        bbox['ymin'] = 0
        bbox['xmax'] = 0
        bbox['ymax'] = 0
        bbox['xlefteye'] = 0
        bbox['ylefteye'] = 0
        bbox['xrighteye'] = 0
        bbox['yrighteye'] = 0
        bbox['xnose'] = 0
        bbox['ynose'] = 0
        bbox['xleftmouth'] = 0
        bbox['yleftmouth'] = 0
        bbox['xrightmouth'] = 0
        bbox['yrightmouth'] = 0
        bbox['x1'] = 0
        bbox['y1'] = 0
        bbox['x2'] = 0
        bbox['y2'] = 0
        bbox['x3'] = 0
        bbox['y3'] = 0
        bbox['x4'] = 0
        bbox['y4'] = 0
        bbox['x5'] = 0
        bbox['y5'] = 0
        bbox['x6'] = 0
        bbox['y6'] = 0
        bbox['x7'] = 0
        bbox['y7'] = 0
        bbox['x8'] = 0
        bbox['y8'] = 0
        bbox['x9'] = 0
        bbox['y9'] = 0
        bbox['x10'] = 0
        bbox['y10'] = 0
        bbox['x11'] = 0
        bbox['y11'] = 0
        bbox['x12'] = 0
        bbox['y12'] = 0
        bbox['x13'] = 0
        bbox['y13'] = 0
        bbox['x14'] = 0
        bbox['y14'] = 0
        bbox['x15'] = 0
        bbox['y15'] = 0
        bbox['x16'] = 0
        bbox['y16'] = 0
        bbox['x17'] = 0
        bbox['y17'] = 0
        bbox['x18'] = 0
        bbox['y18'] = 0
        bbox['x19'] = 0
        bbox['y19'] = 0
        bbox['x20'] = 0
        bbox['y20'] = 0
        bbox['x21'] = 0
        bbox['y21'] = 0
        bbox['x22'] = 0
        bbox['y22'] = 0
        bbox['x23'] = 0
        bbox['y23'] = 0
        bbox['x24'] = 0
        bbox['y24'] = 0
        bbox['x25'] = 0
        bbox['y25'] = 0
        bbox['x26'] = 0
        bbox['y26'] = 0
        bbox['x27'] = 0
        bbox['y27'] = 0
        bbox['x28'] = 0
        bbox['y28'] = 0
        bbox['x29'] = 0
        bbox['y29'] = 0
        bbox['x30'] = 0
        bbox['y30'] = 0
        bbox['x31'] = 0
        bbox['y31'] = 0
        bbox['x32'] = 0
        bbox['y32'] = 0
        bbox['x33'] = 0
        bbox['y33'] = 0
        bbox['x34'] = 0
        bbox['y34'] = 0
        bbox['x35'] = 0
        bbox['y35'] = 0
        bbox['x36'] = 0
        bbox['y36'] = 0
        bbox['x37'] = 0
        bbox['y37'] = 0        
        if len(info) == 6:
            bbox['xmin'] = float(info[2])
            bbox['ymin'] = float(info[3])
            bbox['xmax'] = float(info[4])
            bbox['ymax'] = float(info[5])
        if len(info) == 12:
            bbox['xlefteye'] = float(info[2])
            bbox['ylefteye'] = float(info[3])
            bbox['xrighteye'] = float(info[4])
            bbox['yrighteye'] = float(info[5])
            bbox['xnose'] = float(info[6])
            bbox['ynose'] = float(info[7])
            bbox['xleftmouth'] = float(info[8])
            bbox['yleftmouth'] = float(info[9])
            bbox['xrightmouth'] = float(info[10])
            bbox['yrightmouth'] = float(info[11])
        if len(info)==76:
            bbox['x1'] = float(info[2])
            bbox['y1'] = float(info[3])
            bbox['x1'] = float(info[4])
            bbox['y2'] = float(info[5])
            bbox['x3'] = float(info[6])
            bbox['y3'] = float(info[7])
            bbox['x4'] = float(info[8])
            bbox['y4'] = float(info[9])
            bbox['x5'] = float(info[10])
            bbox['y5'] = float(info[11])
            bbox['x6'] = float(info[12])
            bbox['y6'] = float(info[13])
            bbox['x7'] = float(info[14])
            bbox['y7'] = float(info[15])
            bbox['x8'] = float(info[16])
            bbox['y8'] = float(info[17])
            bbox['x9'] = float(info[18])
            bbox['y9'] = float(info[19])
            bbox['x10'] = float(info[20])
            bbox['y10'] = float(info[21])
            bbox['x11'] = float(info[22])
            bbox['y11'] = float(info[23])
            bbox['x12'] = float(info[24])
            bbox['y12'] = float(info[25])
            bbox['x13'] = float(info[26])
            bbox['y13'] = float(info[27])
            bbox['x14'] = float(info[28])
            bbox['y14'] = float(info[29])
            bbox['x15'] = float(info[30])
            bbox['y15'] = float(info[31])
            bbox['x16'] = float(info[32])
            bbox['y16'] = float(info[33])
            bbox['x17'] = float(info[34])
            bbox['y17'] = float(info[35])
            bbox['x18'] = float(info[36])
            bbox['y18'] = float(info[37])
            bbox['x19'] = float(info[38])
            bbox['y19'] = float(info[39])
            bbox['x20'] = float(info[40])
            bbox['y20'] = float(info[41])
            bbox['x21'] = float(info[42])
            bbox['y21'] = float(info[43])
            bbox['x22'] = float(info[44])
            bbox['y22'] = float(info[45])
            bbox['x23'] = float(info[46])
            bbox['y23'] = float(info[47])
            bbox['x24'] = float(info[48])
            bbox['y24'] = float(info[49])
            bbox['x25'] = float(info[50])
            bbox['y25'] = float(info[51])
            bbox['x26'] = float(info[52])
            bbox['y26'] = float(info[53])
            bbox['x27'] = float(info[54])
            bbox['y27'] = float(info[55])
            bbox['x28'] = float(info[56])
            bbox['y28'] = float(info[57])
            bbox['x29'] = float(info[58])
            bbox['y29'] = float(info[59])
            bbox['x30'] = float(info[60])
            bbox['y30'] = float(info[61])
            bbox['x31'] = float(info[62])
            bbox['y31'] = float(info[63])
            bbox['x32'] = float(info[64])
            bbox['y32'] = float(info[65])
            bbox['x33'] = float(info[66])
            bbox['y33'] = float(info[67])
            bbox['x34'] = float(info[68])
            bbox['y34'] = float(info[69])
            bbox['x35'] = float(info[70])
            bbox['y35'] = float(info[71])
            bbox['x36'] = float(info[72])
            bbox['y36'] = float(info[73])
            bbox['x37'] = float(info[74])
            bbox['y37'] = float(info[75])
            
        data_example['bbox'] = bbox
        dataset.append(data_example)

    return dataset


if __name__ == '__main__':
    dir = '.' 
    net = '24'
    #net='RNet'
    output_directory = 'imglists/RNet'
    run(dir, net, output_directory, shuffling=True)
