import math
import os
import numpy as np
import Bio.PDB
from Bio.PDB import PDBParser, PDBIO, Vector
from ase.io import read, proteindatabank
from abtem import show_atoms
from abtem.transfer import CTF, scherzer_defocus
from abtem import *
from abtem.potentials import *
from abtem.waves import *
from pdbfixer import PDBFixer
from abtem.noise import poisson_noise
import MDAnalysis as mda
from ase import Atoms
from os.path import basename

class Protein:
    """
    Represents a protein structure.

    Args:
        file (str): The path to the PDB file containing the protein structure.
        p1 (int): The first parameter.
        p2 (int): The second parameter.
        p3 (int): The third parameter.

    Attributes:
        file (str): The path to the PDB file containing the protein structure.
        structure (Structure): The parsed protein structure.
        com (ndarray): The center of mass coordinates of the protein structure.

    """

    def __init__(self, file, p1, p2, p3):
        self.file = file
        parser = PDBParser()
        pdb_id = os.path.splitext(os.path.basename(file))[0][:4]  
        self.structure = parser.get_structure(pdb_id, file)
        self.com = self.calculate_com()

    def calculate_com(self):
        """
        Calculates the center of mass (COM) of the protein structure.

        Returns:
            ndarray: The center of mass coordinates.

        """
        total_mass = 0
        mass_coordinates = np.zeros(3)

        for atom in self.structure.get_atoms():
            atom_mass = atom.mass
            mass_coordinates += atom_mass * atom.coord
            total_mass += atom_mass

        com = mass_coordinates / total_mass
        return com

    def read_atoms(self, file, p1, p2, p3):
        """
        Reads the atoms from a PDB file and performs some operations.

        Args:
            file (str): The path to the PDB file.
            p1 (int): The first parameter.
            p2 (int): The second parameter.
            p3 (int): The third parameter.

        Returns:
            atoms: The processed atoms.

        """
        atoms = proteindatabank.read_proteindatabank(file)
        atoms.cell = (p1, p2, p3, 90, 90, 90)
        atoms.center()
        atoms.get_cell_lengths_and_angles()
        show_atoms(atoms, plane='xz')
        show_atoms(atoms, plane='xy')
        show_atoms(atoms, plane='yz')
        return atoms

# Class Protein Manipulator
class ProteinManipulator(Protein):
    """
    A class for manipulating protein structures.

    Inherits from the Protein class.

    Methods:
    - rotate_around_com(rotation_angles): Rotate the protein structure around its center of mass.
    - save_rotated_pdbs(output_folder): Save rotated PDB files with different angles.
    - fix_and_prepare_pdb(): Fix and prepare the PDB file for further analysis.
    """

    def rotate_around_com(self, rotation_angles):
        """
        Rotate the protein structure around its center of mass.

        Args:
        - rotation_angles (list): A list of three angles in degrees representing the rotation around the z-axis,
          rotation around the y-axis, and rotation around the z-axis, respectively.

        """
        rot_z1, rot_y, rot_z2 = map(math.radians, rotation_angles)

        # Create rotation matrices
        rot_matrix_z1 = Bio.PDB.rotaxis(rot_z1, Vector(0, 0, 1))
        rot_matrix_y = Bio.PDB.rotaxis(rot_y, Vector(0, 1, 0))
        rot_matrix_z2 = Bio.PDB.rotaxis(rot_z2, Vector(0, 0, 1))

        for atom in self.structure.get_atoms():
            atom.coord -= self.com
            atom.coord = np.dot(atom.coord, rot_matrix_z1)
            atom.coord = np.dot(atom.coord, rot_matrix_y)
            atom.coord = np.dot(atom.coord, rot_matrix_z2)
            atom.coord += self.com 

    def save_rotated_pdbs(self, output_folder):
        """
        Save rotated PDB files with different angles.

        Args:
        - output_folder (str): The path to the output folder where the rotated PDB files will be saved.

        """
        angle_labels = ('phi', 'theta', 'psi') 
        pdb_id = self.structure.get_id()
        output_subfolder = os.path.join(output_folder, pdb_id)
        os.makedirs(output_subfolder, exist_ok=True)

        step_size = 9  

        original_structure = self.structure.copy() 

        # Iterate through all combinations of angles
        for phi in range(0, 360, step_size):
            for theta in range(0, 360, step_size):
                for psi in range(0, 360, step_size):
                    angles = [phi, theta, psi]  

                    self.structure = original_structure.copy()
                    self.rotate_around_com(angles)

                    file_name_parts = [f"{angle_labels[j]}{angles[j]:06.1f}" for j in range(3)]
                    file_name = "_".join(file_name_parts)
                    output_file = os.path.join(output_subfolder, f"{pdb_id}_{file_name}.pdb")

                    # Save the rotated structure
                    io = PDBIO()
                    io.set_structure(self.structure)
                    io.save(output_file)

        print("All rotated PDB files have been saved.")

    def fix_and_prepare_pdb(self):
        """
        Fix and prepare the PDB file for further analysis.

        This method performs the following steps:
        - Finds missing residues
        - Finds nonstandard residues
        - Replaces nonstandard residues
        - Removes heterogens
        - Finds missing atoms
        - Adds missing atoms
        - Adds missing hydrogens
        - Writes the fixed PDB file

        """
        fixer = PDBFixer(filename=self.file)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(True)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        PDBFile.writeFile(fixer.topology, fixer.positions, open(self.file, 'w+'))

        # Update the structure in the object
        self.structure = Bio.PDB.PDBParser().get_structure(self.structure.id, self.file)


    
class ProteinDiffraction(ProteinManipulator):
    """
    A class for performing protein diffraction and imaging.

    Args:
        file (str): The path to the protein file.
        p1 (int): Parameter 1.
        p2 (int): Parameter 2.
        p3 (int): Parameter 3.
    """

    def __init__(self, file, p1, p2, p3):
        super().__init__(file, p1, p2, p3)

    def prepare_image(self, array):
        """
        Prepare the image array for further processing.

        Args:
            array (numpy.ndarray): The input image array.

        Returns:
            numpy.ndarray: The prepared image array.
        """
        array = (array - array.min()) / array.ptp() * np.iinfo(np.uint16).max
        array = array.astype(np.uint16)
        array = array.T
        return array

    def diffraction_and_imaging(self):
        """
        Perform protein diffraction and imaging.

        Returns:
            tuple: A tuple containing the clean measurement, diffraction data, and noisy measurement.
        """
        u = mda.Universe(self.file)
        #atoms = proteindatabank.read_proteindatabank(self.file)
        positions = u.atoms.positions
        symbols = [a.element for a in u.atoms]
        # Create an ASE Atoms object 
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.set_cell((70, 70, 70, 90, 90, 90))
        atoms.center()
        ctf1 = CTF(energy=200e3)  
        ctf1.defocus = None # Change the defocus value based on specific needs
        ctf2 = CTF(energy=200e3, Cs=-7e-6 * 1e10, focal_spread=10, angular_spread=0.5, gaussian_spread=2)
        ctf2.semiangle_cutoff = None 
        ctf2.defocus = None  # Change the defocus value based on specific needs
        wave = PlaneWave(energy=200e3)
        potential = Potential(atoms, 
                        sampling=0.05, #Change the sampling value based on specific needs
                        parametrization='lobato',
                        slice_thickness=0.5)
        wave.grid.match(potential)
        propagator = FresnelPropagator()
        exit_wave = wave.multislice(potential)
        diffract_wave = exit_wave.diffraction_pattern(max_angle=20, block_zeroth_order=True)
        #print(diffract_wave.shape)
        clean_measurement = self.prepare_image(exit_wave.apply_ctf(ctf1).intensity().array)
        imaging_wave_noisy = exit_wave.apply_ctf(ctf2).intensity()
        noisy_measurement = self.prepare_image(poisson_noise(imaging_wave_noisy, None ).array)
        diffraction = self.prepare_image(diffract_wave.array)

        return clean_measurement, diffraction, noisy_measurement 

    def save_diffraction_and_images(self, output_folder):
        """
        Save the diffraction data and images to the specified output folder.

        Args:
            output_folder (str): The path to the output folder.
        """
        pdb_id = self.structure.get_id()

        projected_image, diffraction_data, noisy_image = self.diffraction_and_imaging()

        diffraction_output_folder = os.path.join(output_folder, '', pdb_id)
        image_output_folder = os.path.join(output_folder, '', pdb_id)
        noisy_output_folder = os.path.join(output_folder, '', pdb_id)

        os.makedirs(diffraction_output_folder, exist_ok=True)
        os.makedirs(image_output_folder, exist_ok=True)
        os.makedirs(noisy_output_folder, exist_ok=True)

        original_filename_without_ext = os.path.splitext(basename(self.file))[0]

        np.save(os.path.join(diffraction_output_folder, f"{original_filename_without_ext}.npy"), diffraction_data)
        np.save(os.path.join(image_output_folder, f"{original_filename_without_ext}.npy"), projected_image)
        np.save(os.path.join(noisy_output_folder, f"{original_filename_without_ext}.npy"), noisy_image)




   

