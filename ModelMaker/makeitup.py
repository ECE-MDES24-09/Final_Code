import argparse
import json
import os
import shutil
from drawitup import drawitup
from cleanitup import cleanitup
from samitup import samitup
from switchitup import switchitup
from trainitup import trainitup
from splititup import splititup


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def get_training_file(dst, src="data.yaml"):
    """
    Copy a file from src to dst directory if it doesn't already exist in dst.

    :param src: Source file path.
    :param dst: Destination directory.
    """
    # Extract filename from source path
    filename = os.path.basename(src)

    # Create full destination path
    dst_path = os.path.join(dst, filename)

    # Check if file already exists in destination
    if not os.path.exists(dst_path):
        shutil.copy(src, dst_path)


def make_it_up(blend_file_path, rotation_step, calc, number_of_renders, iterations, input_dir, output_dir, batch_dir,
               augmentation_options, drawing, cleaning, augment, segment, train, send, split, split_ratios):
    images_filepath = f'{input_dir}/images'
    labels_filepath = f'{input_dir}/labels'

    seg_dir = f"{input_dir}Seg"
    segimages_filepath = f'{seg_dir}/images'
    seglabels_filepath = f'{seg_dir}/labels'
    os.makedirs(seg_dir, exist_ok=True)
    get_training_file(seg_dir)
    training_file = f"Tests.v1i.yolov8/data.yaml"
    aug_options = read_json_file(augmentation_options)


    if drawing:
        print("Generating images")
        drawitup(blend_file_path, rotation_step, calc, number_of_renders, input_dir, iterations)

    if cleaning:
        print("Checking for Corrupt images")
        cleanitup(images_filepath, labels_filepath)


    if segment:
        print("Creating Segmented Dataset")
        samitup(input_dir, seg_dir)


    if augment:
        print("Augmenting Dataset")
        switchitup(seg_dir, input_dir, output_dir, aug_options, split_ratios)

    if train:
        print("Training Model")
        trainitup(training_file)


    if split:
        print("splitting up images")
        splititup(segimages_filepath, seglabels_filepath, batch_dir, 300)









def startup(config):
    # Read JSON configuration
    config = read_json_file(config)

    # Use the values from the JSON file (example)
    blend_file_path = config["blend_file_path"]
    rotation_step = config["rotation_step"]
    calc = config["calc"]
    number_of_renders = config["number_of_renders"]
    iterations = config["iterations"]
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    batch_dir = config["batch_dir"]
    augmentation_options = config["augmentation_options"]
    drawing = config["drawing"]
    cleaning = config["cleaning"]
    augment = config["augment"]
    segment = config["segment"]
    train = config["train"]
    split = config["split"]
    split_ratios = tuple(config["split_ratios"])

    make_it_up(blend_file_path, rotation_step, calc, number_of_renders, iterations, input_dir, output_dir, batch_dir,
               augmentation_options, drawing, cleaning, augment, segment, train, send, split,  split_ratios)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch move images and labels to new directories.")
    parser.add_argument("arg_file", type=str, help="Path to file containing script Options")

    args = parser.parse_args()
    startup(args.arg_file)