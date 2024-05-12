import os
import glob
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib import pyplot 
import argparse

parser = argparse.ArgumentParser(description='Process image data for testing.')
parser.add_argument('--model_path', type=str, help='Path to the trained model file.')
parser.add_argument('--src_folder1', type=str, help='Path to the source folder for diffraction data.')
parser.add_argument('--src_folder2', type=str, help='Path to the source folder for noisy image data.')
parser.add_argument('--target_folder', type=str, help='Path to the target folder for clean image data.')
parser.add_argument('--batch_size', type=int, default=5, help='Batch size for processing images.')
args = parser.parse_args()

# Load the model
g_model = load_model(args.model_path)


def preprocess_image(file_path, is_target, target_size=(256, 256)):
    """Preprocess the image for model input."""
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
        min_val, max_val = np.min(image), np.max(image)
        image = (image - min_val) * (65535.0 / (max_val - min_val))
    image = (image - 32767.5) / 32767.5
    image = np.expand_dims(image, axis=-1)
    return image

def get_all_files(src_folder1, src_folder2, target_folder1):
    """Retrieve all file paths for the dataset."""
    subfolder_names = sorted(os.listdir(src_folder1))
    first_path = os.path.join(src_folder1, subfolder_names[0])

    if os.path.isdir(first_path):
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
    else:
        src_files1 = sorted(glob.glob(os.path.join(src_folder1, '*.npy')))
        src_files2 = sorted(glob.glob(os.path.join(src_folder2, '*.npy')))
        target_files = sorted(glob.glob(os.path.join(target_folder1, '*.npy')))
    return src_files1, src_files2, target_files

def image_generator(src_folder1, src_folder2, target_folder1, batch_size):
    src_files1, src_files2, target_files = get_all_files(src_folder1, src_folder2, target_folder1)
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

def create_dataset(src_folder1, src_folder2, target_folder, batch_size):
    generator, num_files = image_generator(src_folder1, src_folder2, target_folder, batch_size)
    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32)
        )
    )

    return dataset

def generate_fake_samples(g_model, samples1, samples2, patch_shape):
    X = g_model.predict([samples1, samples2])
    return X

def summarize_performance(g_model, dataset, n_samples=5, fig_size=(25, 20), font_size=20):
    X_realA1, X_realA2, X_realB = next(iter(dataset.take(1)))
    X_realA1, X_realA2, X_realB = X_realA1[:n_samples], X_realA2[:n_samples], X_realB[:n_samples]

    X_fakeB = generate_fake_samples(g_model, X_realA1, X_realA2, n_samples)

    
    X_realA1 = (X_realA1 + 1) * 32767.5
    X_realA2 = (X_realA2 + 1) * 32767.5
    X_realB = (X_realB + 1)  * 32767.5
    X_fakeB = (X_fakeB + 1)  * 32767.5

    
    cmap = "gray"  

    fig, axs = pyplot.subplots(4, n_samples, figsize=fig_size)

    params = {'axes.labelsize': font_size, 'axes.titlesize': font_size, 'xtick.labelsize': font_size, 'ytick.labelsize': font_size}
    pyplot.rcParams.update(params)

    for i in range(n_samples):
        ax = axs[0, i]
        ax.set_xticks([0, 128, 255]) 
        ax.set_yticks([0, 128, 255]) 
        ax.imshow(X_realA1[i, :, :, 0], vmin=0, vmax=512, origin='lower')
        ax.set_xlabel('px')
        ax.set_ylabel('px')

    for i in range(n_samples):
        ax = axs[1, i]
        ax.set_xticks([0, 128, 255]) 
        ax.set_yticks([0, 128, 255])
        ax.imshow(X_realA2[i, :, :, 0], origin='lower')
        ax.set_xlabel('px')
        ax.set_ylabel('px')

    for i in range(n_samples):
        ax = axs[2, i]
        ax.set_xticks([0, 128, 255]) 
        ax.set_yticks([0, 128, 255]) 
        ax.imshow(X_fakeB[i, :, :, 0], origin='lower')
        ax.set_xlabel('px')
        ax.set_ylabel('px')

    for i in range(n_samples):
        ax = axs[3, i]
        ax.set_xticks([0, 128, 255]) 
        ax.set_yticks([0, 128, 255]) 
        ax.imshow(X_realB[i, :, :, 0], origin='lower')
        ax.set_xlabel('px')
        ax.set_ylabel('px')

    pyplot.subplots_adjust(wspace=0.1, hspace=0.1)  
    pyplot.tight_layout()
    pyplot.savefig('test_plot.png', dpi=450)
    pyplot.show()
pyplot.close()


if __name__ == "__main__":
    dataset_test = create_dataset(args.src_folder1, args.src_folder2, args.target_folder, args.batch_size)
    summarize_performance(g_model, dataset_test)
