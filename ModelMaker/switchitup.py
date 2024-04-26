import cv2
import numpy as np
from PIL import Image, ImageEnhance, ExifTags
import os
import shutil
from tqdm import tqdm
from numpy import asarray
from sklearn.model_selection import train_test_split
from matplotlib import cm
import random
import logging

# Configure logging
logging.basicConfig(filename='switchitup.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')



def get_bbox_subset(bbox, image):
    length = len(bbox)
    sublist_len = random.randint(0, length)
    bbox_sublist = random.sample(bbox, sublist_len)
    height, width = image.shape[:2]
    bbox_sublist = convert_to_absolute_coordinates(bbox_sublist, width, height)
    return bbox_sublist


def convert_to_absolute_coordinates(labels, image_width, image_height):
    absolute_labels = []
    for label in labels:
        x_center, y_center, width, height = map(float, label)
        x_center, width = x_center * image_width, width * image_width
        y_center, height = y_center * image_height, height * image_height

        # Convert from center coordinates to top-left coordinates
        x = int(x_center - width / 2)
        y = int(y_center - height / 2)
        w = int(width)
        h = int(height)

        absolute_labels.append([x, y, w, h])
    return absolute_labels


def apply_blur(image, pixels, bbox=None):
    if bbox:  # Apply within bounding box
        bbox_sub = get_bbox_subset(bbox, image)
        for box in bbox_sub:
            x, y, w, h = map(int, box)
            try:
                roi = image[y:y + h, x:x + w]
                roi = cv2.blur(roi, (int(pixels), int(pixels)))
                image[y:y + h, x:x + w] = roi
            except Exception as e:
                logging.exception("An error occurred. Could not Perform Bounding Box Annotations")
    else:  # Apply on whole image
        image = cv2.blur(image, (int(pixels), int(pixels)))
    return image


def apply_brightness(image, percent, bbox=None):
    image_pil1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_pil1)
    enhancer = ImageEnhance.Brightness(image_pil)
    ndimage = np.array(image_pil)
    if bbox:  # Apply within bounding box
        bbox_sub = get_bbox_subset(bbox, ndimage)
        for box in bbox_sub:
            x, y, w, h = map(int, box)
            try:
                roi = image_pil.crop((x, y, x + w, y + h))
                enhancer = ImageEnhance.Brightness(roi)
                enhanced_roi = enhancer.enhance(percent / 100)
                image_pil.paste(enhanced_roi, (x, y))
            except Exception as e:
                logging.exception("An error occurred. Could not Perform Bounding Box Annotations")
    else:  # Apply on whole image
        image_pil = enhancer.enhance(percent / 100)

    f_image = np.array(image_pil)
    return cv2.cvtColor(f_image, cv2.COLOR_RGB2BGR)

def apply_crop(image, min_percent, max_percent, bbox=None):
    h, w, channels = image.shape
    min_crop = min_percent / 100
    max_crop = max_percent / 100

    if bbox:  # Apply within bounding box
        bbox_sub = get_bbox_subset(bbox, image)
        for box in bbox_sub:
            x, y, bw, bh = map(int, box)
            try:
                crop_x = int(bw * np.random.uniform(min_crop, max_crop))
                crop_y = int(bh * np.random.uniform(min_crop, max_crop))

                # Cropping and ensuring it stays within image bounds
                cropped_roi = image[max(y + crop_y, 0):min(y + bh - crop_y, h), max(x + crop_x, 0):min(x + bw - crop_x, w)]

                # Creating a black background with the size of original bounding box
                black_bg = np.zeros((bh, bw, channels), dtype=np.uint8)

                # Calculating position to place the cropped image on the black background
                start_y = (bh - cropped_roi.shape[0]) // 2
                start_x = (bw - cropped_roi.shape[1]) // 2

                # Placing the cropped image on the black background
                black_bg[start_y:start_y + cropped_roi.shape[0], start_x:start_x + cropped_roi.shape[1]] = cropped_roi

                # Placing the black background with cropped image onto the original image
                image[y:y + bh, x:x + bw] = black_bg
            except Exception as e:
                logging.exception("An error occurred. Could not Perform Bounding Box Annotations")
    else:  # Apply on whole image
        crop_x = int(w * np.random.uniform(min_crop, max_crop))
        crop_y = int(h * np.random.uniform(min_crop, max_crop))
        image = image[crop_y:h - crop_y, crop_x:w - crop_x]

    return image


def apply_exposure(image, percent, bbox=None):
    exposure_factor = 1 + (percent / 100)
    return apply_brightness(image, exposure_factor, bbox)


def apply_flip(image, horizontal, vertical, bbox=None):
    if bbox:
        bbox_sub = get_bbox_subset(bbox, image)
        for box in bbox_sub:
            x, y, w, h = map(int, box)
            try:
                roi = image[y:y + h, x:x + w]
                if horizontal:
                    roi = cv2.flip(roi, 1)
                if vertical:
                    roi = cv2.flip(roi, 0)
                image[y:y + h, x:x + w] = roi
            except Exception as e:
                logging.exception("An error occurred. Could not Perform Bounding Box Annotations")
    else:
        if horizontal:
            image = cv2.flip(image, 1)
        if vertical:
            image = cv2.flip(image, 0)
    return image


def apply_noise(image, percent, bbox=None):
    noise_level = percent / 100
    if bbox:
        bbox_sub = get_bbox_subset(bbox, image)
        for box in bbox_sub:
            x, y, w, h = map(int, box)
            try:
                roi = image[y:y + h, x:x + w]
                noise = np.random.randint(0, 256, roi.shape, dtype=np.uint8)
                noisy_image = cv2.addWeighted(roi, 1 - noise_level, noise, noise_level, 0)
                image[y:y + h, x:x + w] = noisy_image
            except Exception as e:
                logging.exception("An error occurred. Could not Perform Bounding Box Annotations")
    else:
        noise = np.random.randint(0, 256, image.shape, dtype=np.uint8)
        image = cv2.addWeighted(image, 1 - noise_level, noise, noise_level, 0)
    return image


def apply_rotate(image, degrees, bbox=None):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    if bbox:
        bbox_sub = get_bbox_subset(bbox, image)
        for box in bbox_sub:
            x, y, bw, bh = map(int, box)
            try:
                center = (x + bw // 2, y + bh // 2)
                M = cv2.getRotationMatrix2D(center, degrees, 1.0)
                rotated_roi = cv2.warpAffine(image, M, (w, h))
                image[y:y + bh, x:x + bw] = rotated_roi[y:y + bh, x:x + bw]
            except Exception as e:
                logging.exception("An error occurred. Could not Perform Bounding Box Annotations")
    else:
        M = cv2.getRotationMatrix2D(center, degrees, 1.0)
        image = cv2.warpAffine(image, M, (w, h))

    return image


def apply_shear(image, horizontal, vertical, bbox=None):
    (h, w) = image.shape[:2]

    if bbox:
        bbox_sub = get_bbox_subset(bbox, image)
        for box in bbox_sub:
            x, y, bw, bh = map(int, box)
            try:
                pts1 = np.float32([[x, y], [x + bw, y], [x, y + bh]])
                pts2 = np.float32(
                    [[x + horizontal, y + vertical], [x + bw + horizontal, y + vertical],
                     [x + horizontal, y + bh + vertical]])
                M = cv2.getAffineTransform(pts1, pts2)
                sheared_roi = cv2.warpAffine(image, M, (w, h))
                image[y:y + bh, x:x + bw] = sheared_roi[y:y + bh, x:x + bw]
            except Exception as e:
                logging.exception("An error occurred. Could not Perform Bounding Box Annotations")
    else:
        pts1 = np.float32([[0, 0], [w, 0], [0, h]])
        pts2 = np.float32([[horizontal, vertical], [w + horizontal, vertical], [horizontal, h + vertical]])
        randomx = random.uniform(-0.45, 0.45)
        randomy = random.uniform(-0.45, 0.45)
        M = np.array([[1, randomx, 0],
                                 [randomy, 1, 0]], dtype=np.float32)
        image = cv2.warpAffine(image, M, (w, h))

    return image


def apply_cutout(image, count, percent):
    h, w, _ = image.shape
    for _ in range(count):
        cutout_size = int(np.sqrt(percent / 100 * w * h))
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        x1 = np.clip(x - cutout_size // 2, 0, w)
        y1 = np.clip(y - cutout_size // 2, 0, h)
        x2 = np.clip(x + cutout_size // 2, 0, w)
        y2 = np.clip(y + cutout_size // 2, 0, h)
        image[y1:y2, x1:x2] = 0
    return image


def resize_image(image, target_size):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def apply_mosaic(images):
    max_height = max(image.shape[0] for image in images)
    max_width = max(image.shape[1] for image in images)
    mosaic_image = np.zeros((max_height * 2, max_width * 2, 3), dtype=np.uint8)

    offsets_and_scales = []

    for idx, image in enumerate(images):
        h, w = image.shape[:2]
        scale_x, scale_y = max_width / w, max_height / h
        resized_image = resize_image(image, (max_width, max_height))

        x_offset = (idx % 2) * max_width
        y_offset = (idx // 2) * max_height
        offsets_and_scales.append((x_offset, y_offset, scale_x, scale_y))

        x_start, y_start = x_offset, y_offset
        mosaic_image[y_start:y_start+max_height, x_start:x_start+max_width] = resized_image

    return mosaic_image, offsets_and_scales


def adjust_segmentation_labels(labels, offsets_and_scales):
    adjusted_labels = []
    for label_set, (x_offset, y_offset, scale_x, scale_y) in zip(labels, offsets_and_scales):
        adjusted_set = []
        for label in label_set:
            class_id, *coords = label
            adjusted_coords = []
            for i in range(0, len(coords), 2):  # Process x, y pairs
                x, y = float(coords[i]), float(coords[i + 1])
                x = x * scale_x + x_offset
                y = y * scale_y + y_offset
                adjusted_coords.extend([x, y])
            adjusted_set.append([class_id] + adjusted_coords)
        adjusted_labels.extend(adjusted_set)
    return adjusted_labels



def apply_hue(image, degrees):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image_hsv)
    h = cv2.add(h, degrees)  # Add hue shift
    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return image


def apply_saturation(image, percent):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image_hsv)
    s = s * (1 + percent / 100)  # Increase saturation
    s = np.clip(s, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return image


def apply_ninety(image, clockwise, counter_clockwise, upside_down, bbox):
    if clockwise:
        if bbox:
            apply_rotate(image, 90, bbox)

        # Rotate 90 degrees clockwise
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if counter_clockwise:
        if bbox:
            apply_rotate(image, 270, bbox)
        # Rotate 90 degrees counter-clockwise
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if upside_down:
        if bbox:
            apply_rotate(image, 180, bbox)
        # Rotate 180 degrees to be upside down
        image = cv2.rotate(image, cv2.ROTATE_180)
    return image


# def replicate_and_augment(image, image_pil, bbox, n, augmentations, base_filename):
#     testdir = "Test_Images/Augmented/AugmentedTests/Copy"
#     images = [image]
#     orig_image_pathtest = os.path.join(testdir, f"{base_filename}_orig_test.png")
#     cv2.imwrite(orig_image_pathtest, image)
#     for _ in range(n):
#         augmented_image_orig = image.copy()
#         copy_image_pathtest = os.path.join(testdir, f"{base_filename}_copy_test_cv2.png")
#         cv2.imwrite(copy_image_pathtest, image)
#         augmented_image_pil = image_pil.copy()
#         augmented_image_pil.save(os.path.join(testdir, f"{base_filename}_copy_test_pil.png"))
#         augmented_image = apply_augmentations(augmented_image_orig, augmented_image_pil, bbox, augmentations, base_filename)
#         images.append(augmented_image)
#     return images


def replicate_and_augment(image, image_pil, bbox, n, augmentations, base_filename):
    testdir = "Test_Images/Augmented/AugmentedTests/Copy"
    os.makedirs(testdir, exist_ok=True)

    orig_image_pil = image_pil.copy()
    image_cv2 = np.array(orig_image_pil)
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)
    images = [image_cv2]
    orig_image_pathtest = os.path.join(testdir, f"{base_filename}_orig_test.png")
    cv2.imwrite(orig_image_pathtest, image)

    for _ in range(n):

        augmented_image_pil = image_pil.copy()
        augmented_image_pil.save(os.path.join(testdir, f"{base_filename}_copy_test_pil.png"))

        # Convert augmented PIL image to BGR format for cv2
        augmented_image_cv2 = np.array(augmented_image_pil)
        augmented_image_cv2 = cv2.cvtColor(augmented_image_cv2, cv2.COLOR_RGB2BGR)
        aug_cv2_pathtest = os.path.join(testdir, f"{base_filename}_copy_test_cv2_fixed.png")
        cv2.imwrite(aug_cv2_pathtest, augmented_image_cv2)
        # Apply augmentations to the BGR image
        augmented_image_cv2 = apply_augmentations(augmented_image_cv2, bbox, augmentations,
                                                  base_filename)

        # Convert the augmented BGR image back to RGB for PIL
        augmented_image_pil = cv2.cvtColor(augmented_image_cv2, cv2.COLOR_BGR2RGB)
        augmented_image_pil = Image.fromarray(augmented_image_pil)
        augmented_image_pil.save(os.path.join(testdir, f"{base_filename}_copy_test_pil_final.png"))

        images.append(augmented_image_cv2)  # Append the cv2 image


    return images




def auto_orient(image_path):
    image = Image.open(image_path)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)

    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass

    return image


def contrast_stretching(image):
    # Convert to grayscale for contrast stretching
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply contrast stretching
    min_val = np.percentile(gray, 2)
    max_val = np.percentile(gray, 98)
    stretched = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U,
                              mask=gray > min_val)
    return cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)


def remap_class_names(data, remap_dict):
    # Assuming 'data' is a list of dictionaries with 'class_name' as one of the keys
    for item in data:
        if item['class_name'] in remap_dict:
            item['class_name'] = remap_dict[item['class_name']]
    return data


def tile_image(image, rows, columns):
    tiled_image = np.tile(image, (rows, columns, 1))
    return tiled_image


def adaptive_equalization(image):
    img_y_cr_cb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)

    # apply adaptive equalization on the Y channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y = clahe.apply(y)

    img_y_cr_cb = cv2.merge([y, cr, cb])
    return cv2.cvtColor(img_y_cr_cb, cv2.COLOR_YCrCb2BGR)


def apply_augmentations(image, bbox, augmentations_dict, base_filename):
    testdir = "Test_Images/Augmented/AugmentedTests/"
    # Process preprocessing steps
    if 'preprocessing' in augmentations_dict:
        for preproc, params in augmentations_dict['preprocessing'].items():
            ptemppath = f"{testdir}{preproc}"
            os.makedirs(ptemppath, exist_ok=True)
            # if preproc == 'auto-orient':
            #     image = auto_orient(image)
            if preproc == 'contrast':
                if params['type'] == 'Contrast Stretching':
                    image = contrast_stretching(image)
                if params['type'] == 'Adaptive Equalization':
                    image = adaptive_equalization(image)
            elif preproc == 'tile':
                image = tile_image(image, params['rows'], params['columns'])
            # The remap step is typically applied to labels, not images
            # Implement remap functionality as needed for your labels
            preproc_image_pathtest = os.path.join(ptemppath, f"{base_filename}_{preproc}_test.png")
            cv2.imwrite(preproc_image_pathtest, image)


    # Process augmentations
    if 'augmentation' in augmentations_dict:
        for aug, params in augmentations_dict['augmentation'].items():
            temppath = f"{testdir}{aug}"
            os.makedirs(temppath, exist_ok=True)
            if aug == 'bbblur':
                image = apply_blur(image, params['pixels'], bbox)
            elif aug == 'bbbrightness':
                image = apply_brightness(image, params['percent'], bbox)
            elif aug == 'bbcrop':
                image = apply_crop(image, params['min'], params['max'], bbox)
            elif aug == 'bbexposure':
                image = apply_exposure(image, params['percent'], bbox)
            elif aug == 'bbflip':
                image = apply_flip(image, params['horizontal'], params['vertical'], bbox)
            elif aug == 'bbnoise':
                image = apply_noise(image, params['percent'], bbox)
            elif aug == 'bbninety':
                image = apply_ninety(image, params['clockwise'], params['counter-clockwise'], params['upside-down'],
                                     bbox)
            elif aug == 'bbrotate':
                image = apply_rotate(image, params['degrees'], bbox)
            elif aug == 'bbshear':
                image = apply_shear(image, params['horizontal'], params['vertical'], bbox)
            elif aug == 'blur':
                image = apply_blur(image, params['pixels'])
            elif aug == 'brightness':
                image = apply_brightness(image, params['percent'])
            elif aug == 'crop':
                image = apply_crop(image, params['min'], params['max'])
            elif aug == 'cutout':
                image = apply_cutout(image, params['count'], params['percent'])
            elif aug == 'exposure':
                image = apply_exposure(image, params['percent'])
            elif aug == 'flip':
                image = apply_flip(image, params['horizontal'], params['vertical'])
            elif aug == 'hue':
                image = apply_hue(image, params['degrees'])
            # elif aug == 'mosaic':
            #     # Mosaic implementation would depend on additional images
            #     image = apply_mosaic(image)
            elif aug == 'ninety':
                image = apply_ninety(image, params['clockwise'], params['counter-clockwise'], params['upside-down'], bbox)
            elif aug == 'noise':
                image = apply_noise(image, params['percent'])
            elif aug == 'rotate':
                image = apply_rotate(image, params['degrees'])
            elif aug == 'saturation':
                image = apply_saturation(image, params['percent'])
            elif aug == 'shear':
                image = apply_shear(image, params['horizontal'], params['vertical'])

            aug_image_pathtest = os.path.join(temppath, f"{base_filename}_{aug}_test.png")
            cv2.imwrite(aug_image_pathtest, image)


    return image


def read_labels(file_path):
    boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            box = line.strip().split()
            coords = box[1:]
            boxes.append(coords)
    return boxes


def switchitup(input_dir, bbox_dir, output_dir, augmentations_dict, split_ratios=(0.7, 0.2, 0.1)):
    images_dir = os.path.join(input_dir, 'images')
    labels_dir = os.path.join(input_dir, 'labels')
    bbox_dir = os.path.join(bbox_dir, 'labels')

    # List all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]

    # Split data into train, valid, test
    train_files, test_files = train_test_split(image_files, test_size=1 - split_ratios[0])
    valid_files, test_files = train_test_split(test_files,
                                               test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]))
    all_augmented_images = []
    # Process for each subset
    for subset, files in zip(['train', 'valid', 'test'], [train_files, valid_files, test_files]):
        logging.info(f"Processing {subset} subset")
        subpath = os.path.join(output_dir, subset)
        os.makedirs(subpath,  exist_ok=True)
        subset_img_dir = os.path.join(output_dir, subset, 'images')
        subset_lbl_dir = os.path.join(output_dir, subset, 'labels')
        os.makedirs(subset_img_dir, exist_ok=True)
        os.makedirs(subset_lbl_dir, exist_ok=True)

        for file in tqdm(files):
            try:
                # Read and augment image
                image_path = os.path.join(images_dir, file)
                image_pil = auto_orient(image_path)
                image = asarray(image_pil)



                # Read label file
                base_filename = os.path.splitext(file)[0]
                bbox_path = os.path.join(bbox_dir, base_filename + '.txt')
                bbox = read_labels(bbox_path) if os.path.exists(bbox_dir) else []
                base_filename = os.path.splitext(file)[0]

                augmented_images = replicate_and_augment(image, image_pil, bbox,
                                                         augmentations_dict['augmentation']['image']['versions'],
                                                         augmentations_dict, base_filename)

                # Save original and augmented images

                for i, aug_image in enumerate([image] + augmented_images):
                    aug_image_path = os.path.join(subset_img_dir, f"{base_filename}_{i}.png")
                    cv2.imwrite(aug_image_path, aug_image)

                    # Copy and rename corresponding label file
                    label_filename = base_filename + '.txt'
                    label_path = os.path.join(labels_dir, label_filename)
                    if os.path.exists(label_path):
                        new_label_path = os.path.join(subset_lbl_dir, f"{base_filename}_{i}.txt")
                        shutil.copy(label_path, new_label_path)

                all_augmented_images = [f"{base_filename}_{i}.png" for i in range(len(augmented_images) + 1)]

                logging.info(f"Successfully processed image: {file}")
            except Exception as e:
                logging.exception(f"Error processing image: {file}")

        all_augmented_images += [file for file in files]  # Include original images in the subset for mosaic creation

        if augmentations_dict['augmentation']['mosaic'] == True:
            # Process images in batches of 4 for mosaics
            for i in tqdm(range(0, len(all_augmented_images), 4)):
                try:
                    batch_images = all_augmented_images[i:i + 4]

                    # If the last batch has less than 4 images, skip it
                    if len(batch_images) < 4:
                        continue

                    mosaic_images = [Image.open(os.path.join(subset_img_dir, img)) for img in batch_images]
                    mosaic, offsets_and_scales = apply_mosaic(mosaic_images)

                    # Create and save the mosaic image
                    mosaic_filename = f"mosaic_{i}_{subset}.png"
                    mosaic.save(os.path.join(subset_img_dir, mosaic_filename))

                    # Create and save the corresponding label file
                    mosaic_labels = []
                    for img in batch_images:
                        label_filename = img.replace('.png', '.txt')
                        label_path = os.path.join(subset_lbl_dir, label_filename)
                        if os.path.exists(label_path):
                            with open(label_path, 'r') as file:
                                labels = [line.strip().split() for line in file.readlines()]
                                mosaic_labels.append(labels)

                    adjusted_labels = adjust_segmentation_labels(mosaic_labels, offsets_and_scales)
                    new_label_path = os.path.join(subset_lbl_dir, mosaic_filename.replace('.png', '.txt'))
                    with open(new_label_path, 'w') as file:
                        for label in adjusted_labels:
                            file.write(' '.join(map(str, label)) + '\n')
                except Exception as e:
                    logging.exception("Error creating mosaic")

