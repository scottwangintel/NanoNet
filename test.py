
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
from operator import add
import tensorflow as tf
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
from sklearn.metrics import (
    jaccard_score, f1_score, recall_score, precision_score, accuracy_score, fbeta_score)

from utils import create_dir, load_model_file
from data import load_data

def calculate_metrics(y_true, y_pred):
    y_pred = y_pred > 0.5
    y_pred = y_pred.reshape(-1)
    y_pred = y_pred.astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.reshape(-1)
    y_true = y_true.astype(np.uint8)

    ## Score
    score_jaccard = jaccard_score(y_true, y_pred, average='binary')
    score_f1 = f1_score(y_true, y_pred, average='binary')
    score_recall = recall_score(y_true, y_pred, average='binary')
    score_precision = precision_score(y_true, y_pred, average='binary', zero_division=1)
    score_acc = accuracy_score(y_true, y_pred)
    score_fbeta = fbeta_score(y_true, y_pred, beta=2.0, average='binary', zero_division=1)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_fbeta]

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Load dataset """
    #path = "../../Kvasir-SEG/"
    path = "../../Dataset/Kvasir-SEG/"
    (train_x, train_y), (test_x, test_y) = load_data(path)

    """ Hyperparameters """
    size = (256, 256)
    input_shape = (256, 256, 3)
    model_name = "A"
    model_path = f"files/{model_name}/model.h5"

    """ Directories """
    create_dir(f"results/{model_name}")

    """ Load the model """
    model = load_model_file(model_path)

    """ Sample prediction: To improve FPS """
    image = np.zeros((1, 256, 256, 3))
    mask = model.predict(image)

    """ Testing """
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    dir_path = os.path.join(".", "results", model_name, "masks")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cnt = 0
    for i, (x, y) in enumerate(zip(test_x, test_y)):
        name = y.split("/")[-1].split(".")[0]

        """ Image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        ori_img = image
        image = image/255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        """ Mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        ori_mask = mask
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0
        mask = mask.astype(np.float32)

        """ Time taken """
        start_time = time.time()
        pred_y = model.predict(image)
        total_time = time.time() - start_time
        time_taken.append(total_time)
        print(f"{name}: {total_time:1.5f}")

        """ Metrics calculation """
        score = calculate_metrics(mask, pred_y)
        metrics_score = list(map(add, metrics_score, score))

        """ Saving masks """
        pred_y_saved = pred_y
        pred_y = pred_y[0] > 0.5
        pred_y = pred_y * 255

        pred_y = np.array(pred_y, dtype=np.uint8)

        ori_img = ori_img
        ori_mask = mask_parse(ori_mask)
        pred_y = mask_parse(pred_y)
        sep_line = np.ones((size[0], 10, 3)) * 255

        tmp = [
            ori_img, sep_line,
            ori_mask, sep_line,
            pred_y
        ]

        cat_images = np.concatenate(tmp, axis=1)
        fnn = os.path.join(".", "results", model_name, f"{name}.png")
        print("Write to file name", fnn)
        try:
            success = cv2.imwrite(fnn, cat_images)
            if not success:
                raise Exception(f"Could not write image to {fnn}")
        except Exception as e:
            print(f"Exception occurred: {e}")

        # Assuming pred_y is a binary mask after thresholding
        # and ori_img is the original image read with cv2.IMREAD_COLOR

        # Ensure pred_y_saved is a binary mask with values 0 or 1
        pred_y_saved = pred_y_saved[0] > 0.5

        # Convert the binary mask to uint8 type and scale it to the range [0, 255]
        pred_y_saved = np.array(pred_y_saved, dtype=np.uint8) * 255

        # Create an RGBA image with the original image and an alpha channel
        ori_img_rgba = cv2.cvtColor(ori_img, cv2.COLOR_BGR2BGRA)

        # Choose a color for the mask, e.g., yellow (BGR format)
        mask_color = [0, 255, 255, 128]  # BGR format for yellow and 128 for alpha (semi-transparent)

        # Create a colored mask with the same size as the original image
        colored_mask = np.zeros_like(ori_img_rgba, dtype=np.uint8)

        # Apply the mask color only to the areas where pred_y_saved is True
        mask_indices = np.where(pred_y_saved == 255)
        colored_mask[mask_indices[0], mask_indices[1], :3] = mask_color[:3]  # Apply color to the mask area
        colored_mask[mask_indices[0], mask_indices[1], 3] = mask_color[3]    # Apply alpha to the mask area

        # Superimpose the mask onto the original image
        superimposed_img = cv2.addWeighted(colored_mask, 1, ori_img_rgba, 1, 0)
        # Save the superimposed image to a file
        output_filename = f"{name}_overlay.png"
        output_filepath = os.path.join("results", model_name, output_filename)
        cv2.imwrite(output_filepath, superimposed_img)

        cnt = cnt + 1
        if cnt > 1:
            break

    

    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    f2 = metrics_score[5]/len(test_x)

    print("")
    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f}")

    mean_time_taken = np.mean(time_taken)
    mean_fps = 1/mean_time_taken
    print("Mean FPS: ", mean_fps)
