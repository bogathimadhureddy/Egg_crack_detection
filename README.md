# Egg Crack Detection

## Introduction:
* People are giving priority to high-quality food products. Eggs are excellent sources of protein and nutrients.
* Eggshells are prone to develop cracks during their handling process.
* Egg industries emphasize on eliminating cracked eggs from reaching the consumers, as cracks contaminate the egg contents.
* Automation of crack detection has resulted in the development of acoustic response and computer vision systems.

## Business Problem:
* Manually counting each cracked egg becomes impractical due to the sheer volume, leading to significant time and effort investment. 
* The need for efficiency in food preparation is hindered by the time-consuming nature of manual egg cracking Detection, crucial in meeting customer demands and maintaining a smooth workflow. 
* Human error is a constant risk when manually counting items at scale, potentially resulting in inaccurate inventory records and wastage

## Project architecture:
![image](https://github.com/bogathimadhureddy/Egg_crack_detection/assets/118733598/d1ef8d59-5b3e-474e-b1de-50d0164c90ae)
## Business Objective And Constrains:
### Objective:
* Minimize Manual Counting of Cracked Eggs.
### Constrains:
* Minimize the cost of Solution.
* Minimize the time consuming of egg crack detection.
* Increase the accuracy in prediction of cracked eggs.
```python
!nvidia-smi

import tensorflow as tf
print("GPU is available" if tf.config.list_physical_devices('GPU') else "GPU is NOT available") # Checking the GPU is using or not.

import os
os.getcwd()

!pip install ultralytics

from IPython.display import display, Image
from IPython import display
display.clear_output()

import ultralytics as ut

ut.checks()

! pip install roboflow
display.clear_output()

# Commented out IPython magic to ensure Python compatibility.
!mkdir /content/datasets
# %cd /content/datasets

from roboflow import Roboflow
rf = Roboflow(api_key="ivPEax6pnLar6UOMduGN")
project = rf.workspace("eggcrack").project("egg_crack")
dataset = project.version(1).download("yolov8")

os.getcwd()

"""import shutil

shutil.rmtree('/content/datasets')

import shutil

shutil.rmtree('/content/sample_data')
"""

import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

with open('/content/datasets/Egg_crack-1/data.yaml', 'r') as file:
    data = yaml.safe_load(file)

data

train_path = Path(data['train'])
train_path

all_files = list(train_path.glob('*'))
all_files
display.clear_output()

train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

len(train_files)

len(val_files)

train_folder = Path('/content/datasets/Egg_crack-1/train_new')
train_images_folder = train_folder / 'images'
train_images_folder.mkdir(parents=True, exist_ok=True)
train_labels_folder = train_folder / 'labels'
train_labels_folder.mkdir(parents=True, exist_ok=True)

val_folder = Path('/content/datasets/Egg_crack-1/val')
val_images_folder = val_folder / 'images'
val_images_folder.mkdir(parents=True, exist_ok=True)
val_labels_folder = val_folder / 'labels'
val_labels_folder.mkdir(parents=True, exist_ok=True)

len(os.listdir('/content/datasets/Egg_crack-1/train/images'))

for file in train_files:
    # Assuming label files have the same base name as images but with a different extension
    label_file = Path('/content/datasets/Egg_crack-1/train/labels')/(file.stem + '.txt')

    shutil.move(file, train_images_folder / file.name)

    # Check if the corresponding label file exists before moving
    if label_file.exists():
        shutil.move(label_file, train_labels_folder / (file.stem + '.txt'))

for file in val_files:
    # Assuming label files have the same base name as images but with a different extension
    label_file = Path('/content/datasets/Egg_crack-1/train/labels')/(file.stem + '.txt')

    shutil.move(file, val_images_folder / file.name)

    # Check if the corresponding label file exists before moving
    if label_file.exists():
        shutil.move(label_file, val_labels_folder / (file.stem + '.txt'))

# Update the YAML file with the new paths
data['train'] = str(train_folder)
data['val'] = str(val_folder)


data

# Save the updated YAML file
with open('/content/datasets/Egg_crack-1/data.yaml', 'w') as file:
    yaml.dump(data, file)

from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
#model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train( data='/content/datasets/Egg_crack-1/data.yaml', imgsz=640,epochs=30 , lr0 = 0.00269, lrf= 0.00288, momentum= 0.73375)
  #weight_decay: 0.00015

import shutil
from google.colab import files

# Replace 'source_folder' with the actual name of your source folder
shutil.make_archive("/content/train", 'zip', "/content/runs/detect/train")

# Download the zip file
files.download("/content/train.zip")

Image('/content/runs/detect/train/confusion_matrix.png')

Image('/content/runs/detect/train/confusion_matrix_normalized.png')

Image('/content/runs/detect/train/results.png')

Image('/content/runs/detect/train/val_batch1_labels.jpg')

Image('/content/runs/detect/train/val_batch0_labels.jpg')

Image('/content/runs/detect/train/F1_curve.png')

import shutil
shutil.rmtree('/content/datasets/runs')

import locale
locale.getpreferredencoding = lambda: "UTF-8"
!yolo task=detect mode=val model=/content/runs/detect/train/weights/best.pt data=/content/datasets/Egg_crack-1/data.yaml

"""from google.colab import drive
drive.mount('/content/drive')
"""

!yolo task=detect mode=predict model=/content/runs/detect/train/weights/best.pt source=/content/drive/MyDrive/New_data

from google.colab import drive
drive.mount('/content/drive')

import glob
from IPython.display import display, Image
for image_path in glob.glob(f'/content/runs/detect/predict/*.png')[:9]:
  display(Image(filename = image_path, height = 200))
  print('\n')

import shutil
from google.colab import files

# Replace 'source_folder' with the actual name of your source folder
shutil.make_archive("/content/predicts", 'zip', "/content/runs/detect/predict")

# Download the zip file
files.download("/content/predicts.zip")

project.version(dataset.version).deploy(model_type="yolov8", model_path=f"/content/runs/detect/train")

model = project.version(dataset.version).model

#choose random test set image
import os, random
test_set_loc = "/content/drive/MyDrive/New_data/"
random_test_image = random.choice(os.listdir(test_set_loc))
print("running inference on " + random_test_image)

pred = model.predict(test_set_loc + random_test_image, confidence=40, overlap=30).json()
pred


