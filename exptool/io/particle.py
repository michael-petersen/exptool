"""
particle.py
   driver for input classes and handling particle data

MSP 28 Sep 2021 Original commit
MSP 25 Oct 2021 Tested for compatibility


TODO

"""

from . import psp_io
from . import spl_io


class Input():
    """Input class to adaptively handle various EXP output formats.

    inputs
    ---------------
    filename : str
        the input filename to be read
    comp     : str, optional
        the name of the component for which to extract data. If None, will read primary header and exit.
    legacy   : bool, default=True
        if True, return attributes rather than a dictionary of particle data
    verbose  : int, default 0
        verbosity flag.

    returns
    ---------------
    self        : Input instance
      .header   : dict, all header values pulled from the file
        the .keys() are the names of each component
        each component has a dictionary of values, including 'parameters'
        the details of the force calculation are in 'force'
      .filename : str, the filename that was read
      .comp     : str, name of the component
      .time     : float, the time in the output file
      .data     : dictionary, with keys:
        x       : float, the x position
        y       : float, the y position
        z       : float, the z position
        vx      : float, the x velocity
        vy      : float, the y velocity
        vz      : float, the z velocity
        mass    : float, the mass of the particle
        id   : int, the integer index of the particle
        potE    : float, the potential energy value

      ---or, if legacy=True---

      .xpos     : float, the x position
      .ypos     : float, the y position
      .zpos     : float, the z position
      .xvel     : float, the x velocity
      .yvel     : float, the y velocity
      .zvel     : float, the z velocity
      .mass     : float, the mass of the particle
      .id     : int, the integer index of the particle
      .pote     : float, the potential energy value


    """
    def __init__(self, filename,comp=None, legacy=False,verbose=0): #edited
        """the main driver for the class"""

        # auto-determine the file type
        if 'SPL.' in filename:
            self.style = 'SPL'
        elif 'OUT.':
            self.style = 'OUT'
        else:
            self.style = 'unknown'

        if self.style=='SPL':
            I = spl_io.Input(filename,comp=comp,verbose=verbose)
        elif self.style=='OUT':
            I = psp_io.Input(filename,comp=comp,verbose=verbose)
        else:
            raise ValueError('File type not supported for file "{}"'.format(filename))

        # expose the header
        self.header = I.header


        # what is the ideal legacy error handling?
        if I.comp==None:
            return

        if legacy:
            self.mass = I.data['m']
            self.xpos = I.data['x']
            self.ypos = I.data['y']
            self.zpos = I.data['z']
            self.xvel = I.data['vx']
            self.yvel = I.data['vy']
            self.zvel = I.data['vz']
            self.pote = I.data['potE']

            if (I.header[I.comp]['parameters']['indexing']):
                self.id = I.data['id']

        else:
            self.data = I.data

        self.filename = I.filename
        self.comp     = I.comp
        self.time     = I.time







#
# Below here are helper functions to subdivide and combine particles for parallel processes
#


def convert_psp_to_legacy(PSPInput):
    """helper class to convert to legacy psp_io if needed"""

    PSPOutput = holder()
    PSPOutput.mass = PSPInput.data['m']
    PSPOutput.xpos = PSPInput.data['x']
    PSPOutput.ypos = PSPInput.data['y']
    PSPOutput.zpos = PSPInput.data['z']
    PSPOutput.xvel = PSPInput.data['vx']
    PSPOutput.yvel = PSPInput.data['vy']
    PSPOutput.zvel = PSPInput.data['vz']
    PSPOutput.pote = PSPInput.data['potE']

    #if (I.header[I.comp]['parameters']['indexing']):
    #    PSPOutput.id = I.data['id']
    return PSPOutput



class holder(object):
    '''all the quantities you could ever want to fill in your own PSP-style output.
    '''
    filename = None
    comp = None
    nbodies = None
    time = None
    xpos = None
    ypos = None
    zpos = None
    xvel = None
    yvel = None
    zvel = None
    mass = None
    pote = None
    id   = None





def subdivide_particles_list(ParticleInstance,particle_roi):
    '''fill a new array with particles that meet this criteria
    '''
    holder = holder()
    holder.xpos = ParticleInstance.xpos[particle_roi]
    holder.ypos = ParticleInstance.ypos[particle_roi]
    holder.zpos = ParticleInstance.zpos[particle_roi]
    holder.xvel = ParticleInstance.xvel[particle_roi]
    holder.yvel = ParticleInstance.yvel[particle_roi]
    holder.zvel = ParticleInstance.zvel[particle_roi]
    holder.mass = ParticleInstance.mass[particle_roi]
    holder.filename = ParticleInstance.filename
    holder.comp = ParticleInstance.comp
    holder.nbodies = ParticleInstance.nbodies
    holder.time = ParticleInstance.time
    return holder




def mix_particles(ParticleInstanceArray):
    '''flatten arrays from multiprocessing into one ParticleInstance.
    '''
    n_instances = len(ParticleInstanceArray)
    n_part = 0
    for i in range(0,n_instances):
        n_part += len(ParticleInstanceArray[i].xpos)
    final_holder = holder()
    final_holder.xpos = np.zeros(n_part)
    final_holder.ypos = np.zeros(n_part)
    final_holder.zpos = np.zeros(n_part)
    final_holder.xvel = np.zeros(n_part)
    final_holder.yvel = np.zeros(n_part)
    final_holder.zvel = np.zeros(n_part)
    final_holder.mass = np.zeros(n_part)
    final_holder.pote = np.zeros(n_part)
    #holder.filename = ParticleInstance.filename
    #holder.comp = ParticleInstance.comp
    #holder.nbodies = ParticleInstance.nbodies
    final_holder.time = ParticleInstanceArray[0].time # only uses first time, should be fine?
    #
    #
    first_part = 0
    for i in range(0,n_instances):
        n_instance_part = len(ParticleInstanceArray[i].xpos)
        final_holder.xpos[first_part:first_part+n_instance_part] = ParticleInstanceArray[i].xpos
        final_holder.ypos[first_part:first_part+n_instance_part] = ParticleInstanceArray[i].ypos
        final_holder.zpos[first_part:first_part+n_instance_part] = ParticleInstanceArray[i].zpos
        final_holder.xvel[first_part:first_part+n_instance_part] = ParticleInstanceArray[i].xvel
        final_holder.yvel[first_part:first_part+n_instance_part] = ParticleInstanceArray[i].yvel
        final_holder.zvel[first_part:first_part+n_instance_part] = ParticleInstanceArray[i].zvel
        final_holder.mass[first_part:first_part+n_instance_part] = ParticleInstanceArray[i].mass
        final_holder.pote[first_part:first_part+n_instance_part] = ParticleInstanceArray[i].pote
        first_part += n_instance_part
    return final_holder


def _create_particle_blocks(nmax,nsplit):
    """randomly partition indices into a set number of groups"""

    # set the random seed for reproducibility
    np.random.seed(1)

    # dummy array of indices
    all_indices = np.arange(0,nmax,1)

    # shuffle the dummy array (in place)
    np.random.shuffle(all_indices)

    # now split up however many times, giving the last array the remainder
    npersplit = int(np.floor(nmax/nsplit))

    IndexList = dict()
    for i in range(0,nsplit-1):
        IndexList[i] = all_indices[i*npersplit:(i+1)*npersplit]

    # give remainder to last array
    IndexList[nsplit-1] = all_indices[(nsplit-1)*npersplit:]

    return IndexList
