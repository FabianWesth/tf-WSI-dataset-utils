import tensorflow as tf
import glob
import os
from os import listdir
from os.path import isfile, join
import random
from dataset_util import get_random_wsi_patch, save_wsi_thumbnail_mask

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_immediate_files(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith(".svs")]


def input_setup(
    batch_size=1, dataset_dir="/home/fabian/UKE_projects/CPC/data/AIH_DILI/liver_cleaned/WSI/AIH/", wsi_ext=".svs"
):

    """
    This function sets up the input data pipeline
    takes input at tfrecord files
    """

    #######################################################################

    # data A input pipeline
    # wsi_paths = list(glob('{}*{}'.format(dataset_dir, wsi_ext)))
    # wsi_paths = [str(path) for path in wsi_paths]

    wsi_paths = get_immediate_files(dataset_dir)
    wsi_paths = [join(dataset_dir, path) for path in wsi_paths]

    for path in wsi_paths:
        save_wsi_thumbnail_mask(path)

    random.shuffle(wsi_paths)
    image_count = len(wsi_paths)
    print("found: {} images from WSI dataset".format(image_count))

    # setup tf dataset using py_function
    path_ds = tf.data.Dataset.from_tensor_slices(wsi_paths)
    path_ds = path_ds.repeat(100)

    def get_random_wsi_patch_lambda(filename):
        return tf.py_function(get_random_wsi_patch, [filename], tf.float32)

    wsi_dataset = path_ds.map(lambda filename: get_random_wsi_patch_lambda(filename), num_parallel_calls=40)
    # wsi_dataset = path_ds.map(tf.py_function(get_random_wsi_patch_lambda, tf.float32), num_parallel_calls=40)
    wsi_dataset = wsi_dataset.shuffle(100)
    wsi_dataset = wsi_dataset.batch(batch_size=batch_size, drop_remainder=True)
    wsi_dataset = wsi_dataset.prefetch(buffer_size=1)  # <-- very important for efficency
    # iterator = tf.data.Iterator.from_structure(wsi_dataset.output_types, wsi_dataset.output_shapes)
    # training_init_op = iterator.make_initializer(wsi_dataset)
    # get_input = iterator.get_next()

    # for wsi in wsi_dataset:
    #     print(wsi)

    wsi_dataset_iter = iter(wsi_dataset)
    pos_batch = wsi_dataset_iter.get_next()

    """
    to use run:
        "sess.run([training_init_op])"

    "get_input"     will return the next batch of images when any part of the
                    TF-graph that relies on it is called

    for more info see:
            https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/data/Dataset

    """


if __name__ == "__main__":
    input_setup()
