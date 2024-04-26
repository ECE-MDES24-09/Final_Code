import comet_ml
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from ultralytics import YOLO

import os

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Restrict TensorFlow to only allocate a specific amount of memory on the first GPU
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

experiment = Experiment(
  api_key="noN1pXsdgwkmeG9OIwoffbFLO",
  project_name="Board&Button2",
  workspace="jboyle1013"
)

os.environ["COMET_EVAL_LOG_CONFUSION_MATRIX"] = "true"
os.environ['COMET_EVAL_BATCH_LOGGING_INTERVAL'] = "2"

def trainitup(training_file):
    # Create a new YOLO model from scratch
    model = YOLO('yolov8s-seg.yaml')  # build from YAML and transfer weights

    # Train the model for 3 epochs
    results = model.train(data=training_file, batch=-1, epochs=10, verbose=True,
                          visualize=True, plots=True, dnn=True, project='Board&Button2')


    # Evaluate the model's performance on the validation set
    results1 = model.val()

    results2 = model("img.png")  # predict on an image

    # Export the model to ONNX format
    success = model.export(format='onnx')
