import os
import tensorflow as tf
import keras_cv
import pandas as pd
from tensorflow import keras
# pylint: disable=no-member


class Preprocess():
    """_Helps preprocessing the data to train and validate ."""


    def __init__(self,image_path,json_path,btch_size,class_dict):
        """Initializes the object and saves the parameter in member variables

        Args:
            image_path (str): path where the images are present
            json_path (str): path of the json file
            btch_size (int): bacth size of the data to be traiend
            class_dict (dict): dictioanry of the classes to encode the label values
        """

        self.image_path = image_path
        self.json_path = json_path
        self.class_dict = class_dict
        self.btch_size = btch_size
        self.image_paths = []
        self.bbox = []
        self.classes = []

    def parse_bdd(self):
        """Parses the BDD100k dataset to match the YOLOV8 input format

        Returns:
            tuple{list,list,list): returns a list of image_paths, bounding boxes and class labels
        """

        data = pd.read_json(self.json_path)
        data = data.apply(self.parse_rows, axis=1)

        return self.image_paths, self.bbox, self.classes

    def parse_rows(self, row):
        """Parse rows of dataframe to fill member variables image_paths, bbox, classes.

        Args:
            row (pandas.core.series.Series): row of a dataframe
        """

        image_file_path = os.path.join(self.image_path, row["name"])
        class_names = []
        bbox_xyxys = []

        if isinstance(row["labels"], list):
            for box in row["labels"]:
                if box["category"] in self.class_dict.keys():
                    
                    class_name = self.class_dict[box["category"]]
                    bbox_xyxy = [float(box["box2d"]["x1"]), float(box["box2d"]["y1"]), float(box["box2d"]["x2"]), float(box["box2d"]["y2"])] 
                    class_names.append(class_name)
                    bbox_xyxys.append(bbox_xyxy)

        self.image_paths.append(image_file_path)
        self.bbox.append(bbox_xyxys)
        self.classes.append(class_names)

    def get_tf_data(self):
        """Creates tf data from parsed input data

        Returns:
            tensorflow.python.data.ops.batch_op._BatchDataset: tf data 
        """

        image_paths, bbox, classes = self.parse_bdd()
        bbox = tf.ragged.constant(bbox)
        classes = tf.ragged.constant(classes)
        image_paths = tf.ragged.constant(image_paths)
        data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))
        data = data.map(self.load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
        data = data.shuffle(self.btch_size * 4)
        data = data.ragged_batch(self.btch_size, drop_remainder=True)
        return data

    def dict_to_tuple(self, inputs):
        """Converts dictionary to tuple

        Args:
            inputs (dict): dict containing input images and bounding boxes

        Returns:
            tuple[tensorflow.python.framework.ops.Tensor,dict]: tuple of the input data
        """

        return inputs["images"], inputs["bounding_boxes"]
    
    def load_image(self,image_path):
        """Load image to infer

        Args:
            image_path (tensorflow.python.framework.ops.Tensor): image path to be loaded

        Returns:
            tensorflow.python.framework.ops.Tensor: loaded image
        """
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        return image
    
    def load_dataset(self, image_path, classes, bbox):
        """Loads dataset for training,reads images

        Args:
            image_path (tensorflow.python.framework.ops.Tensor): path of the image
            classes (tensorflow.python.framework.ops.Tensor): encoded label values
            bbox (tensorflow.python.ops.ragged.ragged_tensor.RaggedTensor): bbox positions

        Returns:
            dict: _description_
        """

        image = self.load_image(image_path)

        bounding_boxes = {
            "classes": tf.cast(classes, dtype=tf.float32),
            "boxes": bbox,
        }
    
        return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}

    def augmenter(self):
        """Augmenter for training

        Returns:
            keras.src.engine.sequential.Sequential: sequential object
        """

        return keras.Sequential(
        layers=[
            keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xyxy"),
            keras_cv.layers.JitteredResize(
                target_size=(640, 640),
                scale_factor=(1.0, 1.0),
                bounding_box_format="xyxy",
            ),
        ]
    )

    def resizing(self):
        """Resizing augmenter for validation

        Returns:
            keras_cv.layers.preprocessing.jittered_resize.JitteredResize: JitteredResize object
        """

        return  keras_cv.layers.JitteredResize(
        target_size=(640, 640),
        scale_factor=(1.0, 1.0),
        bounding_box_format="xyxy",
    )