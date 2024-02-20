import keras_cv
# pylint: disable=not-callable


class YOLOV8:
    """Creates  a keras_cv model object"""

    def __init__(self,class_mapping):
        """Initializes YOLOV8 object and created keras_cv backbone and Detector object

        Args:
            class_mapping (dict): dictionary of class labels and integers
        """
        self.backbone = keras_cv.models.YOLOV8Backbone.from_preset(
            "yolo_v8_l_backbone_coco",
            load_weights=True
        )
        self.class_mapping = class_mapping
        self.model = keras_cv.models.YOLOV8Detector(
            num_classes=len(class_mapping),
            bounding_box_format="xyxy",
            backbone=self.backbone,
            fpn_depth=3,
        )
        
    def metrics_display(self,data):
        """Display the metrics of the model in the passed data

        Args:
            data (tensorflow.python.data.ops.prefetch_op._PrefetchDataset): tf daat to be evaluated

        Returns:
            dict: metrics dict with BoxCOCO metrics
        """
        metrics = keras_cv.metrics.BoxCOCOMetrics(
                bounding_box_format="xyxy",
                evaluate_freq=1e9,
            )
        
        count=0 
        for batch in data:
            print("Calculating metrics for ",count,end='\r')
            images, y_true = batch[0], batch[1]
            y_pred = self.model.predict(images, verbose=0)
            metrics.update_state(y_true, y_pred)
            print(metrics.result(force=True),end='\r')
            count+=1

        # print(type())
        metrics_val = metrics.result()
        for key,val in metrics_val.items():
            print(key ," : ",val)
        return metrics_val
