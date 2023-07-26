
import numpy as np
import argparse

try:
    import h5py
except ImportError:
    raise ImportError("You will need to 'pip install h5py --user' to use this converter.")

from . import particle

class HDFConverter():
    def __init__(self, filename,comp=None,verbose=0):
        """the main driver"""
        self.filename = filename

        convert_psp_to_hdf5(self.filename)
        # if a component is selected, perhaps we should only convert that?

    @staticmethod
    def convert_psp_to_hdf5(inputfilename):
        """
        """
        # define file
        outputfilename = inputfilename + '.h5'

        # steps: open file, get all components
        O = particle.Input(inputfilename)
        comps = list(O.header.keys())

        # open the new file for printing
        f = h5py.File(outputfilename, 'w')

        # not using a global header, so add some attributes here
        f['time'] = O.time

        for comp in comps:

            f.create_group(comp)

            HDFConverter.print_component_header(f,O,comp)

            HDFConverter.make_phasespace(f,inputfilename,comp)

        f.close()


    @staticmethod
    def print_component_header(f,O,comp):
        f[comp].create_group('header')
        for key in O.header[comp].keys():
            try: # see if there is another dictionary level
                for subkey in O.header[comp][key].keys():
                    try: # see if there is another dictionary level
                        for subsubkey in O.header[comp][key][subkey].keys(): # this is the deepest we ever have to go
                            try:
                                f['{}/header/{}/{}'.format(comp,key,subkey)].attrs.create(subsubkey,O.header[comp][key][subkey][subsubkey])
                                #print('{}/header/{}/{}'.format(comp,key,subkey),subsubkey,O.header[comp][key][subkey][subsubkey])
                            except:
                                f['{}/header/{}'.format(comp,key)].create_group(subkey)
                                f['{}/header/{}/{}'.format(comp,key,subkey)].attrs.create(subsubkey,O.header[comp][key][subkey][subsubkey])
                                #print('CREATE ','{}/header/{}/{}'.format(comp,key,subkey),subsubkey,O.header[comp][key][subkey][subsubkey])
                    except:
                        try:
                            f['{}/header/{}'.format(comp,key)].attrs.create(subkey,O.header[comp][key][subkey])
                            #print('{}/header/{}'.format(comp,key),subkey,O.header[comp][key][subkey])
                        except:
                            f['{}/header'.format(comp)].create_group(key)
                            f['{}/header/{}'.format(comp,key)].attrs.create(subkey,O.header[comp][key][subkey])
                            #print('CREATE ','{}/header/{}'.format(comp,key),subkey,O.header[comp][key][subkey])
            except:
                f['{}/header'.format(comp)].attrs.create(key,O.header[comp][key])
                #print('{}/header'.format(comp),key,O.header[comp][key])

    @staticmethod
    def make_phasespace(f,inputfilename,comp):
        # get the data
        O1 = particle.Input(inputfilename,comp)
        # make the phase space
        PS = np.array([O1.data['m'],O1.data['x'],O1.data['y'],O1.data['z'],O1.data['vx'],O1.data['vy'],O1.data['vz'],O1.data['potE']]).T
        # print the phase space
        dset = f[comp].create_dataset('phasespace', data=PS)


"""
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
