import os
import shutil
from tqdm import tqdm


def create_batch_directories(base_dir, batch_num):
    batch_dir = os.path.join(base_dir, f'batch_{batch_num}')
    images_dir = os.path.join(batch_dir, 'images')
    labels_dir = os.path.join(batch_dir, 'labels')
    os.makedirs(batch_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    return images_dir, labels_dir


def splititup(images_dir, labels_dir, base_dest_dir, batch_size):
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    batch_num = 1

    for i, image_file in tqdm(enumerate(image_files, start=1), total=len(image_files), desc="Processing Files"):
        label_file = os.path.splitext(image_file)[0] + '.txt'

        if i % batch_size == 1:
            # Create new batch directories for images and labels
            images_batch_dir, labels_batch_dir = create_batch_directories(base_dest_dir, batch_num)
            batch_num += 1

        # Move image and its corresponding label file to the appropriate batch directory
        shutil.copy(os.path.join(images_dir, image_file), os.path.join(images_batch_dir, image_file))
        if os.path.exists(os.path.join(labels_dir, label_file)):
            shutil.copy(os.path.join(labels_dir, label_file), os.path.join(labels_batch_dir, label_file))


if __name__ == '__main__':
    # Base directory containing 'images' and 'labels'
    base_directory = 'dataWBoardColors1Seg/train'

    # Destination base directory
    dest_base_directory = 'db1batches'

    # Process images and labels
    splititup(os.path.join(base_directory, 'images'),
              os.path.join(base_directory, 'labels'),
              dest_base_directory, 300)
