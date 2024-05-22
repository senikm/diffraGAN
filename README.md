

# DiffraGAN: Phasing of single molecule diffraction data using low-resolution frames and high resolution diffraction data

DiffraGAN is designed to facilitate the phasing of single molecule diffraction data, utilizing low-resolution frames and high resolution diffraction data. This repository includes tools for manipulating protein `.pdb` files obtained from the Protein Data Bank (PDB), generate image, diffraction pattern pairs using AbTEM simulation package, and train conditional generative adversarial network for phasing. 

## Features
- **Protein manipulation**: Rotate, fix, and diffract `.pdb` files using our main `protein.py` script.
- **Model Training**: Train/retrain DiffraGAN with provided training files, which include preprocessing steps and model details. These may be updated based on the data requirements.


## Environment setup
This project requires specific Python packages to run. We recommend setting up a Conda environment using the provided `environment.yml` file.

### Creating the environment
To create the Conda environment with the necessary packages, run the following command:

```bash
conda env create -f environment.yml
```
We recommend installing `TensorFlow` in a separate environment to avoid potential conflicts with other dependencies, especially if GPU support is required. 


# Test model performance

The `test.py`  script evaluates the performance of a trained model using diffraction, noisy, and clean image datasets.

## Requirements
Ensure you have Python and the necessary packages installed, including TensorFlow, NumPy, OpenCV, and Matplotlib.

## Downloading pretrained weights and test data

We are currently training the next version of the DiffraGAN which will be served through Hugging Face. Meanwhile, you can download pretrained weights of the smaller model and test data required to run the model from the following links:

- **Pretrained weights and test data**: [Download pretrained weights and test data](https://drive.google.com/drive/folders/1DthFnK07dIKVrFCZB8QUSnGJPoJ6BRrI?usp=sharing)

Please note that the current model has been trained on assymetric proteins up to 40 kDa and has limited capabilities. After downloading, please ensure that you extract the files.

## Usage

Run the script by providing paths to the required folders and the model file. 

```bash
python test.py --model_path 'path/to/model.h5' \
               --src_folder1 'path/to/diffraction_data' \
               --src_folder2 'path/to/noisy_image_data' \
               --target_folder 'path/to/clean_image_data' \
               --batch_size 5
```

## Ongoing development and updates

 This repository will continue to serve as the main source for ongoing development related to the DiffraGAN. It will include future updates and model changes, ensuring that improvements and advancements are shared with the community.

## To-do list

- [ ] Denoising layer.
- [ ] Better integration of diffraction data.
- [ ] Wide range of training data and simulation parameters.


## Citation

If you find our code useful for your research, please, consider citing our paper:

```bibtex

@article{Matinyan2024,
  author = {Matinyan, Senik, Filipcik, Pavel, van Genderen, Eric and Abrahams, Jan Pieter},
  title = {DiffraGAN: A conditional generative adversarial network for phasing single molecule diffraction data to atomic resolution},
  journal = {Frontiers in Molecular Biosciences},
  volume = {11},
  pages = {1386963},
  year = {2024},
  doi = {10.3389/fmolb.2024.1386963}
}
```