#!/usr/bin/env python

import os
import argparse
from protein import ProteinDiffraction
import glob


def process_pdb_files_in_folder(rotated_folder, output_folder, p1, p2, p3):
    for protein_id_dir in glob.glob(os.path.join(rotated_folder, "*")):
        dir_name = os.path.basename(protein_id_dir)
        #print(dir_name)
        if dir_name in ['']:
        #print(f"Processing {dir_name}")
            for pdb_file in glob.glob(os.path.join(protein_id_dir, "*.pdb")):
                # Initialize Protein Diffraction object
                protein = ProteinDiffraction(pdb_file, p1, p2, p3)
                # Save diffraction, projected image and noisy images triplets
                protein.save_diffraction_and_images(output_folder)
            else:
                pass



def main():
    parser = argparse.ArgumentParser(description='Process PDB files in train and test folders.')
    parser.add_argument('train_folder', default = '', nargs='?', help='Path to the train folder containing PDB files')
    parser.add_argument('test_folder', default = '', nargs='?', help='Path to the test folder containing PDB files')
    parser.add_argument('--p1', type=int, default=70, help='p1 unit cell parameter for ProteinManipulator')
    parser.add_argument('--p2', type=int, default=70, help='p2 unit cell parameter for ProteinManipulator')
    parser.add_argument('--p3', type=int, default=70, help='p3 unit cell parameter for ProteinManipulator')

    args = parser.parse_args()

    train_input_folder = os.path.join(args.train_folder, '')
    test_input_folder = os.path.join(args.test_folder, '')

    process_pdb_files_in_folder(train_input_folder, args.train_folder, args.p1, args.p2, args.p3)
    process_pdb_files_in_folder(test_input_folder, args.test_folder, args.p1, args.p2, args.p3)

if __name__ == '__main__':
    main()


