import json
import tensorflow as tf
import matplotlib.pyplot as plt
from preprocessing import Preprocess
from yolov8_detecter import YOLOV8

MODEL_PATH = "model_val.h5"

def draw_bar_plot(dict_name,image_name):
    plt.figure(figsize=(17, 10))
    plt.bar(list(dict_name.keys()), list(dict_name.values()), color=["#f575a6", "#f795c7", "#f9cbe7", "#c4b5e0", "#85aadf" ,"#447ac4"])
    plt.yticks(fontsize=20) 
    plt.savefig(image_name)

if __name__ == "__main__":

    # Load the json
    config = json.load(open("config.json"))

    # Map the labels to encode in preprocessing
    class_mapping = dict(zip(range(len(config["classes"])), config["classes"]))
    class_dict = dict(zip(config["classes"], range(len(config["classes"]))))
    
    # Preprocessing steps for validation data
    val_preprocess = Preprocess(config["val"]["data_path"], config["val"]["annot_path"],\
                        config["batch_size"], class_dict)
    val_data = val_preprocess.get_tf_data()
    resizing = val_preprocess.resizing()
    val_ds = val_data.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(val_preprocess.dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # Create model object and load weights
    yolo = YOLOV8(class_mapping)
    yolo.model.load_weights(MODEL_PATH)

    # # Display the metrics
    metrics = yolo.metrics_display(val_ds)

    # Convert it into dictionary
    metrics_dict = {k:tf.keras.backend.get_value(v) for k, v in metrics.items()}

    # Seperate mAP and recall for plotting
    map_dict = {}
    recall_dict = {}
    for key,val in metrics_dict.items():
        if "MaP" in key:
            map_dict[key] = round(val,2)
        if "Recall" in key:
            recall_dict[key] = round(val,2)
   
    # Plot the graphs for visualization
    
    draw_bar_plot(map_dict, "val_mAP.png")
    draw_bar_plot(recall_dict, "val_recall.png")


# Final results 
    
# {'MaP': 0.13673218, 'MaP@[IoU=50]':0.26028723, 
#  'MaP@[IoU=75]':0.122820914, 'MaP@[area=small]': 0.07464098, 
#  'MaP@[area=medium]': 0.21730292, 'MaP@[area=large]': 0.2398369,
#    'Recall@[max_detections=1]': 0.1047412, 'Recall@[max_detections=10]': 0.18807389, 
#    'Recall@[max_detections=100]':0.19195154, 'Recall@[area=small]': 0.11215344, 
#  'Recall@[area=medium]': 0.27871448, 'Recall@[area=large]': 0.29837514}