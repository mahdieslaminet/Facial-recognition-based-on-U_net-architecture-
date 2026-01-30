
import os
import cv2
import time
import numpy as np
import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger

from unet import build_unet
from glob import glob

global IMAGE_HEIGHT
global IMAGE_WIDTH
global INPUT_SHAPE
global EPOCHS_COUNT
global new_directories
global DATASET_PATH

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
DATASET_PATH = "dataset\\Thyroid Dataset\\tg3k"

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 1)
EPOCHS_COUNT = 10

new_directories = [
        f'{DATASET_PATH}\\train\\images',
        f'{DATASET_PATH}\\train\\labels',
        f'{DATASET_PATH}\\val\\images',
        f'{DATASET_PATH}\\val\\labels',
        f'{DATASET_PATH}\\test\\images',
        f'{DATASET_PATH}\\test\\labels'
        ]

def read_image_mask(x, y, exapnd_it = False):
    """ Image """
    x = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (IMAGE_WIDTH, IMAGE_HEIGHT))
    
    """ Mask """
    y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    y = cv2.resize(y, (IMAGE_WIDTH, IMAGE_HEIGHT))

    if exapnd_it:
        x = x/255.0
        x = x.astype(np.float32)

        y = y/255.0
        y = y.astype(np.float32)

        x = np.expand_dims(x, axis=-1) # 256 X 256 X 1
        y = np.expand_dims(y, axis=-1)
    
    return x, y

def _image_to_tensor(func, x, y):
    image, mask = tf.numpy_function(func, [x, y], [tf.float32, tf.float32])
     
    image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    mask.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    return image, mask

def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()
        return read_image_mask(x, y, True)

    image, mask = _image_to_tensor(f, x, y)

    return image, mask

def tf_dataset(X, Y, batch=8):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.shuffle(buffer_size=5000).map(preprocess) 
    ds = ds.batch(batch).prefetch(2)
    return ds

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path):

    train_x = sorted(glob(os.path.join(path, "train", "images", "*.jpg")))
    train_y = sorted(glob(os.path.join(path, "train", "labels", "*.jpg")))

    valid_x = sorted(glob(os.path.join(path, "val", "images", "*.jpg")))
    valid_y = sorted(glob(os.path.join(path, "val", "labels", "*.jpg")))

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.jpg")))
    test_y = sorted(glob(os.path.join(path, "test", "labels", "*.jpg")))

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def change_dataset():
    import json

    for new_directory in new_directories:
        create_dir(new_directory)

    with open(f'{DATASET_PATH}\\tg3k-trainval.json') as f:
        data = json.load(f)
        for i, file_info in enumerate(data['val']):
            file_name = f'{file_info}.jpg'
            img_path = f'{new_directories[0]}\\{file_name}'
            mask_path = f'{new_directories[1]}\\{file_name}'

            if i % 7 == 0:
                os.rename(img_path, f"{new_directories[2]}\\{file_name}")
                os.rename(mask_path, f"{new_directories[3]}\\{file_name}")
            else:
                os.rename(img_path, f"{new_directories[4]}\\{file_name}")
                os.rename(mask_path, f"{new_directories[5]}\\{file_name}")


def train():
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Loading the dataset """
    (train_x, train_y), (valid_x, valid_y), (_, _) = load_dataset(DATASET_PATH)

    """ Dataset Pipeline """
    train_ds = tf_dataset(train_x, train_y)
    valid_ds = tf_dataset(valid_x, valid_y)

    """ Directory for storing files """
    create_dir("files")
    model_name = 'UNet'
    model_info = f'{model_name}_{len(train_x)}imgs_{EPOCHS_COUNT}epochs11'
    model_path = os.path.join("files", f"model_{model_info}.h5")
    csv_path = os.path.join("files", f"data_{model_info}.csv")

    print(f"Train: {len(train_x)} - Valid: {len(valid_x)}")
    print("")

    """ Model """
    unet_model = build_unet(INPUT_SHAPE)

    unet_model.compile(
        loss=[sm.losses.binary_crossentropy],
        metrics=[sm.metrics.iou_score],
        optimizer=tf.keras.optimizers.Adam(1e-4)
    )

    """ Training """
    callbacks = [
            ModelCheckpoint(model_path, verbose=1, save_best_only=True, monitor='val_loss'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=1e-7, verbose=1),
            CSVLogger(csv_path, append=True),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=False)
            ]

    start_time = time.time()
    unet_model.fit(train_ds,
                   validation_data=valid_ds,
                   epochs=EPOCHS_COUNT,
                   callbacks=callbacks)


# Test

def test():
    import os
    import numpy as np
    import tensorflow as tf
    
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Hyperparameters """

    """ Paths """
    model_name = 'model_UNet_3226imgs_10epochs.h5'
    model_path = os.path.join("files", model_name)

    """ Directory for storing files """
    dirc_name = f'{model_name}_results'
    create_dir(dirc_name)

    """ Loading the dataset """
    (_, _), (_, _), (test_x, test_y) = load_dataset(DATASET_PATH)

    print(f"Test: {len(test_x)}/{len(test_x)}")
    print("")

    """ Load the model """
    model = tf.keras.models.load_model(model_path, compile=False)

    """ Prediction & Evaluation """
    i = 0
    for x, y in zip(test_x, test_y):
        print(i)
        i += 1
        """ Extract the name """
        name = x.split("\\")[-1].split(".")[0]

        """ Reading the image and mask"""
        image, mask = read_image_mask(x, y)
        h, w = image.shape
        image_x = image.copy()
        image = image/255.0
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0) ## [1, H, W]

        """ Prediction """
        pred = model.predict(image, verbose=0)[0]

        pred = (pred > 0.5) * 255
        pred = pred.astype(np.int32)
        pred = cv2.resize(pred, (w, h))

        line = np.ones((image_x.shape[0], 10)) * 255

        image_x = image_x.astype(np.int32)
        pred = pred.astype(np.int32)

        cat_images = np.concatenate([image_x, line, mask, line, pred], axis=1)
        cv2.imwrite(f'{dirc_name}//{name}.jpg', cat_images)

test()
