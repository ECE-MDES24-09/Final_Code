import os
from PIL import Image
from tqdm import tqdm

def is_image_corrupt(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()  # verify that it is, in fact, an image
    except (IOError, SyntaxError):
        return True
    return False


def delete_corresponding_files(image_path, label_path):
    try:
        os.remove(image_path)
        os.remove(label_path)
        # print(f"Deleted {image_path} and {label_path}")
    except OSError as e:
        print(f"Error deleting files: {e}")


def cleanitup(images_dir, labels_dir):
    for filename in tqdm(os.listdir(images_dir)):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # check for image file extensions
            image_path = os.path.join(images_dir, filename)
            label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + '.txt')

            if is_image_corrupt(image_path):
                delete_corresponding_files(image_path, label_path)
