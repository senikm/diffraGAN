import glob
import os
import numpy as np
import cv2
import tensorflow as tf

def preprocess_image(file_path, is_target, target_size=(256, 256)):
    image = np.load(file_path)
    if is_target:
        padded_image = np.zeros((256, 256), dtype=image.dtype)
        pad_top = (1024 - image.shape[0]) // 2
        pad_left = (1024 - image.shape[1]) // 2
        padded_image[pad_top:pad_top+image.shape[0], pad_left:pad_left+image.shape[1]] = image
        image = padded_image
    else:
        #Interplote the image to the target size: INTER_LANCZOS4 for upscaling, INTER_AREA for downscaling
        interpolation = cv2.INTER_LANCZOS4 if image.shape[0] < target_size[0] or image.shape[1] < target_size[1] else cv2.INTER_AREA
        image = cv2.resize(image, target_size, interpolation=interpolation)
        min_val = np.min(image)
        max_val = np.max(image)
        image = (image - min_val) * (65535.0 / (max_val - min_val)) 

    # Normalize image based on the range of the data you have 
    image = (image - 32767.5) / 32767.5

    # Add channel dimension
    image = np.expand_dims(image, axis=-1)
    return image

def image_generator(src_folder1, src_folder2, target_folder1, batch_size):
    subfolder_names = sorted(os.listdir(src_folder1))
    src_files1, src_files2, target_files = [], [], []
    for subfolder in subfolder_names:
        src_subfolder1 = os.path.join(src_folder1, subfolder)
        src_subfolder2 = os.path.join(src_folder2, subfolder)
        target_subfolder = os.path.join(target_folder1, subfolder)

        src_files_subfolder1 = sorted(glob.glob(os.path.join(src_subfolder1, '*.npy')))
        src_files_subfolder2 = sorted(glob.glob(os.path.join(src_subfolder2, '*.npy')))
        target_files_subfolder = sorted(glob.glob(os.path.join(target_subfolder, '*.npy')))

        src_files1.extend(src_files_subfolder1)
        src_files2.extend(src_files_subfolder2)
        target_files.extend(target_files_subfolder)

    num_files = len(src_files1)

    def generator():
        while True:
            shuffled_index_source = np.random.permutation(num_files)
            src_files_shuffled1 = [src_files1[i] for i in shuffled_index_source]
            src_files_shuffled2 = [src_files2[i] for i in shuffled_index_source]
            target_files_shuffled = [target_files[i] for i in shuffled_index_source]
            for i in range(0, num_files - batch_size, batch_size):  
                batch_start = i
                batch_end = min(i + batch_size, num_files)
                src_batch1 = [preprocess_image(file, False) for file in src_files_shuffled1[batch_start:batch_end]]
                src_batch2 = [preprocess_image(file, False) for file in src_files_shuffled2[batch_start:batch_end]]
                target_batch = [preprocess_image(file, False) for file in target_files_shuffled[batch_start:batch_end]]
                yield np.array(src_batch1), np.array(src_batch2), np.array(target_batch)
    return generator, num_files

def create_dataset(src_folder1, src_folder2, target_folder1, batch_size):
    generator, num_files = image_generator(src_folder1, src_folder2, target_folder1, batch_size)
    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32)
        )
    )
    return dataset
