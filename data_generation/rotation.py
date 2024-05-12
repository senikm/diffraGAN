#!/usr/bin/env python

import os
import argparse
from protein import ProteinManipulator
import glob



def process_pdb_files_in_folder(folder_path, output_folder, p1, p2, p3):

    used_pdbs_path = os.path.join(folder_path, 'used_pdbs')

    for pdb_file_path in glob.glob(os.path.join(used_pdbs_path, '*.pdb')):
        
        print(f"Processing {pdb_file_path}")

        # Initialize ProteinManipulator object
        protein = ProteinManipulator(pdb_file_path, p1, p2, p3)

        # Fix and prepare PDB
        protein.fix_and_prepare_pdb()

        # Rotate and save PDBs
        protein.save_rotated_pdbs(output_folder)

        # Additional processing can be added here
        print(f"Finished processing {pdb_file_path}")


def main():
    parser = argparse.ArgumentParser(description='Process PDB files in train and test folders.')
    parser.add_argument('train_folder', default = '', nargs='?', help='Path to the train folder containing PDB files')
    parser.add_argument('test_folder', default = '', nargs='?', help='Path to the test folder containing PDB files')
    parser.add_argument('--p1', type=int, default=70, help='p1 unit cell parameter for ProteinManipulator')
    parser.add_argument('--p2', type=int, default=70, help='p2 unit cell parameter for ProteinManipulator')
    parser.add_argument('--p3', type=int, default=70, help='p3 unit cell  parameter for ProteinManipulator')

    args = parser.parse_args()

    train_output_folder = os.path.join(args.train_folder, '')
    test_output_folder = os.path.join(args.test_folder, '')

    process_pdb_files_in_folder(args.train_folder,train_output_folder, args.p1, args.p2, args.p3)
    process_pdb_files_in_folder(args.test_folder, test_output_folder, args.p1, args.p2, args.p3)

if __name__ == '__main__':
    main()




