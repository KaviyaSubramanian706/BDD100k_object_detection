# Object Detection on BDD 100k Dataset

YoloV8 object detection model training and inference.

## Description

This project helps training ,evaluating and visualizing the yolov8 object detection model.

## Getting Started


### Installing

* Create a virtual environment
* Clone the git repository
* Move inside the cloned folder and install requirements.txt
```
pip install -r requirements.txt
```

### Executing program

* To train the model:
* Download the data from Berekely official site and give the data and json location in config.json
* Change the model_name in config.json ,so that model will be saved in that name.
* Then execute
```
python train.py
```
* To visualize the dataset
* Enter the option to visualize either ground truth or predictions
* Download the pretrained model from one drive link https://drive.google.com/file/d/1AMXPVSnpwCy3FPaF0HD36U_5ARwKB5ur/view?usp=sharing and paste in the current folder
```
python visualize.py
```
* To display metrics
```
python metrics_display.py
```
This displays BoxCOCO evaluation metrics and saves images with names val_mAP.png and val_recall.png

