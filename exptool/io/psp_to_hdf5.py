"""
draft conversion from PSP format the HDF5.


For each component group, there is a subgroup named 'header', which stores header information related to that component. This information may include various parameters and metadata. The header information is organized into nested groups and attributes within the 'header' subgroup. The structure of the header data may vary depending on the specific PSP file format.

The main data associated with each component is stored in a dataset named 'phasespace' within the component group. This dataset is an Nx8 array, where N represents the number of particles. Each row of the dataset corresponds to a particle and contains the following information in this order: mass, x-coordinate, y-coordinate, z-coordinate, x-velocity, y-velocity, z-velocity, and potential energy.


Example usage:
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog='PSP2HDF5',
                        description='Convert OUTPSN files to HDF5 format.')#epilog='Text at the bottom of help')

    parser.add_argument('filename',help='the file to be converted')

    args = parser.parse_args()

    HDFConverter.convert_psp_to_hdf5(args.filename)



# example usage
# to run, do PSP2HDF5 OUT.run0.00000

import h5py
outputfilename = 'OUT.run0.00000.h5'
outputfilename = 'OUT.run0.00000.h5'

# how is the global header information saved?
# only time is saved
f = h5py.File(outputfilename, 'r')

# if the component is called 'halo', then
f['halo/phasespace'] # is the dataset, (Nx8), with (mass,x,y,z,vx,vy,vz,potential) for each particle

# the header data is saved in a group,
f['halo/header']
# and then each group may have subgroups or attributes. typically, you will have
f['halo/header/parameters']
print(f['halo/header/parameters'].attrs.keys())
print(f['halo/header/parameters'].attrs['nEJwant'])

f['halo/header/force']
print(f['halo/header/force'].attrs['id'])
print(f['halo/header/force/parameters'].attrs.keys())
for key in f['halo/header/force/parameters'].attrs.keys():
    print(key,f['halo/header/force/parameters'].attrs[key])


"""



import numpy as np
import h5py
from . import particle

class HDFConverter():
    """
    HDFConverter class for converting custom PSP (Particle Simulation Program) files to HDF5 format.

    This class allows you to convert data from PSP format into HDF5 format, which is a versatile and efficient data storage format.

    Parameters:
        filename (str): The name of the PSP input file to be converted.
        comp (str): Optional. The specific component to convert. If provided, only the data for the specified component will be converted.
        verbose (int): Optional. Verbosity level for printing progress and messages during conversion.

    Attributes:
        filename (str): The name of the PSP input file.
    """

    def __init__(self, filename, comp=None, verbose=0):
        """
        Initialize the HDFConverter instance.

        Args:
            filename (str): The name of the PSP input file to be converted.
            comp (str, optional): The specific component to convert. If provided, only the data for the specified component will be converted.
            verbose (int, optional): Verbosity level for printing progress and messages during conversion.
        """
        self.filename = filename
        self.comp = comp
        self.verbose = verbose

        # Start the conversion process
        self.convert_psp_to_hdf5()

    @staticmethod
    def convert_psp_to_hdf5(inputfilename):
        """
        Convert a PSP input file to HDF5 format.

        Args:
            inputfilename (str): The name of the PSP input file to be converted.
        """
        # Define the output file name
        outputfilename = inputfilename + '.h5'

        # Open the PSP input file and extract components
        O = particle.Input(inputfilename)
        comps = list(O.header.keys())

        # Create a new HDF5 file for storing the converted data
        f = h5py.File(outputfilename, 'w')

        # Store the simulation time as an attribute
        f['time'] = O.time

        for comp in comps:
            # Create a group for each component
            f.create_group(comp)

            # Print the header information for the component
            HDFConverter.print_component_header(f, O, comp)

            # Create and store the phase space data for the component
            HDFConverter.make_phasespace(f, inputfilename, comp)

        # Close the HDF5 file
        f.close()

    @staticmethod
    def print_component_header(f, O, comp):
        """
        Print header information for a component to an HDF5 file.

        Args:
            f (h5py.Group): The HDF5 group to store the header information.
            O (particle.Input): The PSP input object.
            comp (str): The name of the component.
        """
        f[comp].create_group('header')
        for key in O.header[comp].keys():
            # Check for nested dictionary levels
            try:
                for subkey in O.header[comp][key].keys():
                    # Check for further nested levels
                    try:
                        for subsubkey in O.header[comp][key][subkey].keys():
                            # Create attributes for the deepest level
                            try:
                                f['{}/header/{}/{}'.format(comp, key, subkey)].attrs.create(subsubkey, O.header[comp][key][subkey][subsubkey])
                            except:
                                # Create subgroups if necessary
                                f['{}/header/{}/{}'.format(comp, key, subkey)].create_group(subsubkey)
                                f['{}/header/{}/{}'.format(comp, key, subkey)].attrs.create(subsubkey, O.header[comp][key][subkey][subsubkey])
                    except:
                        # Create attributes for the intermediate level
                        try:
                            f['{}/header/{}/{}'.format(comp, key, subkey)].attrs.create(subkey, O.header[comp][key][subkey])
                        except:
                            # Create subgroups if necessary
                            f['{}/header'.format(comp)].create_group(key)
                            f['{}/header/{}'.format(comp, key)].attrs.create(subkey, O.header[comp][key][subkey])
            except:
                # Create attributes for the top-level header
                f['{}/header'.format(comp)].attrs.create(key, O.header[comp][key])

    @staticmethod
    def make_phasespace(f, inputfilename, comp):
        """
        Convert and store phase space data for a component in an HDF5 file.

        Args:
            f (h5py.Group): The HDF5 group to store the phase space data.
            inputfilename (str): The name of the PSP input file.
            comp (str): The name of the component.
        """
        # Read data from the PSP input file
        O1 = particle.Input(inputfilename, comp)

        # Create a phase space array
        PS = np.array([O1.data['m'], O1.data['x'], O1.data['y'], O1.data['z'], O1.data['vx'], O1.data['vy'], O1.data['vz'], O1.data['potE']]).T

        # Store the phase space data as a dataset
        dset = f[comp].create_dataset('phasespace', data=PS)
