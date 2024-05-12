import numpy as np
from matplotlib import pyplot
import os
from model_details import discriminator, generator, cgan
from data_processing import create_dataset, image_generator



def predict_samples(g_model, samples1, samples2, patch_shape):
    X = g_model.predict([samples1, samples2])
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

def summarize_performance(step, g_model, d_model, dataset, n_samples=5):
    X_realA1, X_realA2, X_realB = next(iter(dataset.take(1)))
    X_realA1, X_realA2, X_realB = X_realA1[:n_samples], X_realA2[:n_samples], X_realB[:n_samples]
    X_fakeB, _ = predict_samples(g_model, X_realA1, X_realA2, n_samples)
    X_realA1 = (X_realA1 + 1) * 32767.5
    X_realA2 = (X_realA2 + 1) * 32767.5
    X_realB = (X_realB + 1)  * 32767.5
    X_fakeB = (X_fakeB + 1)  * 32767.5
    for i in range(n_samples):
        pyplot.subplot(4, n_samples, 1 + i)
        #Define the tick_params() function in a way you prefer
        pyplot.tick_params(labelbottom=False, labelleft=False)
        pyplot.imshow(X_realA1[i, :, :, 0], vmin=0, vmax=1024)
    for i in range(n_samples):
        pyplot.subplot(4, n_samples, 1 + n_samples + i)
        pyplot.tick_params(labelbottom=False, labelleft=False)
        pyplot.imshow(X_realA2[i, :, :, 0])
    for i in range(n_samples):
        pyplot.subplot(4, n_samples, 1 + n_samples * 2 + i)
        pyplot.tick_params(labelbottom=False, labelleft=False)
        pyplot.imshow(X_fakeB[i, :, :, 0])
    for i in range(n_samples):
        pyplot.subplot(4, n_samples, 1 + n_samples * 3 + i)
        pyplot.tick_params(labelbottom=False, labelleft=False)
        pyplot.imshow(X_realB[i, :, :, 0])
    filename1 = 'plot_%06d.png' % (step+1)
    pyplot.savefig(filename1, dpi=450)
    pyplot.close()
    filename2 = 'g_model_%06d.h5' % (step+1)
    filename3 = 'd_model_%06d.h5' % (step+1)
    g_model.save(filename2)
    d_model.save(filename3)
    print('>Saved: %s and %s' % (filename1, filename2))

def train(d_model, g_model, gan_model, src_folder1, src_folder2, target_folder1, n_epochs=200, n_batch=16):
    n_patch = d_model.output_shape[1]
    d_losses1, d_losses2, g_losses = [], [], []
    generator_func, num_files = image_generator(src_folder1, src_folder2, target_folder1, n_batch)
    generator = generator_func()
    steps_per_epoch = num_files // n_batch
    for epoch in range(n_epochs):
        for step in range(steps_per_epoch):
            X_realA1, X_realA2, X_realB = next(generator)
            y_real = np.ones((n_batch, n_patch, n_patch, 1))
            X_fakeB, y_fake = predict_samples(g_model, X_realA1, X_realA2, n_patch)
            d_loss1 = d_model.train_on_batch([X_realA1,X_realA2, X_realB], y_real)
            d_loss2 = d_model.train_on_batch([X_realA1,X_realA2, X_fakeB], y_fake)
            g_loss, _, _ = gan_model.train_on_batch([X_realA1, X_realA2], [y_real, X_realB])
            d_losses1.append(d_loss1)
            d_losses2.append(d_loss2)
            g_losses.append(g_loss)
        if (epoch + 1) % 10 == 0:
            dataset_temp = create_dataset(src_folder3, src_folder4, target_folder2, batch_size=64)
            summarize_performance(epoch, g_model, d_model, dataset_temp)
    pyplot.plot(d_losses1, label='Discriminator Loss - Real', color='blue')
    pyplot.plot(d_losses2, label='Discriminator Loss - Fake', color='magenta')
    pyplot.plot(g_losses, label='Generator Loss', color='red')
    pyplot.xlabel('Batch')
    pyplot.ylabel('Loss')
    pyplot.legend()
    pyplot.savefig('training_losses.png', dpi=450)



# define image shape
image_shape = (256,256, 1)

# load image data
# Train data
src_folder1 = ''
src_folder2 = ''
target_folder1 = ''

#Test data
src_folder3 = ''
src_folder4 = ''
target_folder2 = ''
d_model = discriminator(image_shape)
g_model = generator(image_shape)

# Define the composite model
gan_model = cgan(g_model, d_model, image_shape)

generator_weights_path =''
discriminator_weights_path =''

# Load discriminator weights
if os.path.exists(discriminator_weights_path):
    d_model.load_weights(discriminator_weights_path)
    print("Loaded discriminator weights")
else:
    print("Wrong path D")

# Load generator weights
if os.path.exists(generator_weights_path):
    g_model.load_weights(generator_weights_path)
    print("Loaded generator weights")
else:
    print("Wrong path G")
# train model
train(d_model, g_model, gan_model, src_folder1, src_folder2, target_folder1)


