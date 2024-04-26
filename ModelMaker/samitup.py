import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchvision
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())


def show_mask(mask, ax, random_color=False, color=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif color:
        color = np.concatenate([np.array(color), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis('on')
    plt.show()


def show_points_on_image(raw_image, input_points, input_labels=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
        labels = np.ones_like(input_points[:, 0])
    else:
        labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis('on')
    plt.show()


def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
        labels = np.ones_like(input_points[:, 0])
    else:
        labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis('on')
    plt.show()


def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
        labels = np.ones_like(input_points[:, 0])
    else:
        labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis('on')
    plt.show()


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_masks_on_image(raw_image, masks, scores):
    if len(masks.shape) == 4:
        masks = masks.squeeze()
    if scores.shape[0] == 1:
        scores = scores.squeeze()

    nb_predictions = scores.shape[-1]
    fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))

    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask = mask.cpu().detach()
        axes[i].imshow(np.array(raw_image))
        show_mask(mask, axes[i])
        axes[i].title.set_text(f"Mask {i + 1}, Score: {score.item():.3f}")
        axes[i].axis("off")
    plt.show()


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis('on')
    plt.show()


def getLabels(labelPath):
    with open(labelPath) as f:
        # Preparing list for annotation of BB (bounding boxes)
        labels = []
        for line in f:
            labels += [line.rstrip()]

    return labels


def readLabelBB(labels, w, h):
    parsedLabels = []
    for i in range(len(labels)):
        bb_current = labels[i].split()
        objClass = bb_current[0]
        x_center, y_center = int(float(bb_current[1]) * w), int(float(bb_current[2]) * h)
        box_width, box_height = int(float(bb_current[3]) * w), int(float(bb_current[4]) * h)
        parsedLabels.append((x_center, y_center, box_width, box_height))
    return parsedLabels, objClass


def getConvertedBoxes(labels, image_width, image_height):
    converted_boxes = []
    class_ids = []
    for i in range(len(labels)):
        bb_current = labels[i].split()
        class_id = int(bb_current[0])
        x_center, y_center = float(bb_current[1]), float(bb_current[2])
        box_width, box_height = float(bb_current[3]), float(bb_current[4])

        # Convert to top left and bottom right coordinates
        x0 = int((x_center - box_width / 2) * image_width)
        y0 = int((y_center - box_height / 2) * image_height)
        x1 = int((x_center + box_width / 2) * image_width)
        y1 = int((y_center + box_height / 2) * image_height)
        class_ids.append(class_id)
        converted_boxes.append([x0, y0, x1, y1])
    return class_ids, converted_boxes

def samitup(bbox_dataset, seg_dataset):
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    # if you would like to plot and view the segmentation masks then set Objects list from the yaml file
    Objects = []
    # color rgb values for each class
    color = []
    # iterate through each image file and add it to a tuple
    image_labels = []

    image_files = glob.glob(f"./{bbox_dataset}/**/*.png", recursive=True)
    print("Number of images:", len(image_files))
    for imgPath in image_files:
        # get the label file path
        labelPath = imgPath.replace(".png", ".txt")
        # rplace images with labels
        labelPath = labelPath.replace("images", "labels")
        # add the image and label path to a tuple
        image_labels.append((imgPath, labelPath))

    for objects in Objects:
        # create a random color and add it to the color list
        color.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))

    loopCount = 0
    for imgPath, labelPath in tqdm(image_labels):
        destination = f'{seg_dataset}'
        # if label file is in destination folder then skip
        label_file = imgPath.split('/')[-1].split('.')[0]
        seg_label_path = os.path.join(destination, f'labels/{label_file}.txt')
        if os.path.exists(seg_label_path):
            label_file = imgPath.split('/')[-1].split('.')[0]
            # print(f'{label_file} already exists in {destination}')
            continue
        labels = getLabels(labelPath)
        image = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        predictor.set_image(image)
        raw_image = Image.open(imgPath).convert("RGB")
        h, w = image.shape[:2]
        class_ids, bounding_boxes = getConvertedBoxes(labels, w, h)
        # show_boxes_on_image(raw_image, bounding_boxes)
        input_boxes = torch.tensor(bounding_boxes, device=predictor.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        for i, mask in enumerate(masks):
            binary_mask = masks[i].squeeze().cpu().numpy().astype(np.uint8)
            contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            try:
                largest_contour = max(contours, key=cv2.contourArea)
                segmentation = largest_contour.flatten().tolist()
                mask = segmentation

                # convert mask to numpy array of shape (N,2)
                mask = np.array(mask).reshape(-1, 2)

                # normalize the pixel coordinates
                mask_norm = mask / np.array([w, h])
                class_id = class_ids[i]
                yolo = mask_norm.reshape(-1)
                # show_mask(mask.cpu().numpy(), plt.gca(), random_color=False, color=color[class_id])
                # check if train or valid in imagPath

                # if folder does not exist, create it
                if not os.path.exists(destination):
                    os.makedirs(destination)
            except Exception as e:
                continue
            # label file name
            loopCount += 1

            # print(f'writing {label_file} to {destination}')
            # print(f"file number {loopCount}")
            # create labels folder if it does not exist
            if not os.path.exists(os.path.join(destination, 'labels')):
                os.makedirs(os.path.join(destination, 'labels'))
            with open(seg_label_path, "a") as f:
                for val in yolo:
                    f.write("{} {:.6f}".format(class_id, val))
                f.write("\n")

        # create images folder if it does not exist
        if not os.path.exists(os.path.join(destination, 'images')):
            os.makedirs(os.path.join(destination, 'images'))
        # copy image to destination/images
        shutil.copy(imgPath, f'{destination}/images')
        # for box in input_boxes:
        #     show_box(box.cpu().numpy(), plt.gca())
        # plt.axis('off')
        # plt.show()
        # if loopCount == 10:
        #     break




