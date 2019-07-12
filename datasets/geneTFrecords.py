import os
import tensorflow as tf
from datasets import xml2TFRecords
"""
usage:
python geneTFrecords.py --xml_img_txt_path=./logs/train_xml.txt --output_dir=tfrecords 
                        --output_name=annotated_data --samples_per_files=2000
"""

tf.app.flags.DEFINE_string(
    'output_name', 'icdar15_annotated_data',
    'Basename used for TFRecords output files.'
)
tf.app.flags.DEFINE_string(
    'output_dir', 'tfrecords',
    'Output directory where to store TFRecords files.'
)
tf.app.flags.DEFINE_string(
    'xml_img_txt_path', './logs/train_xml.txt',
    'the path forward to the txt which contain the info about the path of images and the gt xml file associated'
)
tf.app.flags.DEFINE_integer(
    'samples_per_files', 2000,
    'setting one tf_record contains how many pictures'
)
FLAGS = tf.app.flags.FLAGS

def main(_):
    if not FLAGS.xml_img_txt_path or not os.path.exists(FLAGS.xml_img_txt_path):
        raise ValueError('You must supply a dataset directory with parameter  --xml_img_txt_path')
    print('Dataset directory:', FLAGS.xml_img_txt_path)
    print('Output directory:', FLAGS.output_dir)

    xml2TFRecords.run(FLAGS.xml_img_txt_path, FLAGS.output_dir, FLAGS.output_name, FLAGS.samples_per_files)

if __name__ == '__main__':
    tf.app.run()