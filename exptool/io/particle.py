"""
# new scaffolding for reading: this will be a driver.

MSP 28 Sep 2021 Original commit


TODO
 - make file selection automatic

"""

from . import psp_io
from . import spl_io


class Input():
    """Input class that wraps psp_io and spl_io to have uniform behaviour."""
    def __init__(self, filename,comp=None, legacy=True,verbose=0):

        if 'SPL' in filename:
            spl = True
        else:
            spl = False
        
        if spl:
            I = spl_io.Input(filename,comp=comp,verbose=verbose)
        else:
            I = psp_io.Input(filename,comp=comp,verbose=verbose)

        if legacy:
            self.mass = I.data['m']
            self.xpos = I.data['x']
            self.ypos = I.data['y']
            self.zpos = I.data['z']
            self.xvel = I.data['vx']
            self.yvel = I.data['vy']
            self.zvel = I.data['vz']
            self.pote = I.data['potE']
            try: # I think this can be done with the header flags instead of try?
                self.indx = I.data['index']
            except:
                pass
        else:
            self.data = I.data

        self.filename = I.filename
        self.comp     = I.comp
        self.time     = I.time
        

        
def revert_to_legacy(I):
        """routine to make the dictionary style from above a drop-in replacement for old psp_io"""

        O = holder()
        
        O.mass = I.data['m']
        O.xpos = I.data['x']
        O.ypos = I.data['y']
        O.zpos = I.data['z']
        O.xvel = I.data['vx']
        O.yvel = I.data['vy']
        O.zvel = I.data['vz']
        O.pote = I.data['potE']
        O.filename = I.filename
        O.comp = I.comp
        O.time = I.time
            
        try:
            O.indx = I.data['index']
        except:
            pass


        return O
                




#
# Below here are helper functions to subdivide and combine particles for parallel processes
#

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





def subdivide_particles_list(ParticleInstance,particle_roi):
    '''fill a new array with particles that meet this criteria
    '''
    holder = particle_holder()
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
    final_holder = particle_holder()
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


