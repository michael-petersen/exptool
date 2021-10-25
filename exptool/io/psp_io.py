"""
Python PSP (Phase-Space Protocol) reader

MSP 25 Oct 2014 in original form
MSP  3 Dec 2015 committed to exptool
MSP  7 Mar 2016 constructed to theoretically handle niatr/ndatr
MSP 27 Aug 2016 added compatibility for dictionary support, the long-term goal of the reader once I commit to re-engineering everything.
MSP  8 Dec 2016 cleaned up subdividing inputs. needs much more cleaning, particularly eliminating many 'self' items from the Input class. Should also set up dictionary dump by default, could just engineer in at the end?
MSP 11 Mar 2019 set up to read yaml-derived input files. A method to diagnose problems would be amazing--currently written elsewhere.
MSP 14 Aug 2019 handle indexing=True from exp component inputs
MSP 17 Dec 2019 major revision to simplify
MSP 28 Sep 2021 deprecate parallelisms (move to particle.py)
MSP 25 Oct 2021 streamline, align with spl_io for merge

PSP is a file format used by the EXP basis function expansion N-body code
written by Martin Weinberg.


TODO
-add handling for multiple components simultaneously (maybe)
-add handling for reading in parts of files

"""

import numpy as np

try:
    # requires yaml support: likely needs to be installed.
    import yaml
except ImportError:
    raise ImportError("You will need to 'pip install pyyaml' to use this reader.")
    

class Input:
    """Input class to adaptively handle OUT. format specifically

    inputs
    ---------------
    filename : str
        the input filename to be read
    comp     : str, optional
        the name of the component for which to extract data. If None, will read primary header and exit.
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
        index   : int, the integer index of the particle
        potE    : float, the potential energy value

    """
    def __init__(self, filename,comp=None,verbose=0):
        """the main driver"""
        self.verbose  = verbose       
        self.filename = filename
        
        # initial check for file validity
        try:
            self.f = open(self.filename, 'rb')
        except Exception:
            raise IOError('Failed to open "{}"'.format(filename))

        # do an initial read of the header
        self.primary_header = dict()

        # initialise dictionaries
        self.comp_map       = dict()
        self.header      = dict()
       
        self._read_primary_header()

        self.comp = comp
        _comps = list(self.header.keys())
 
        # if a component is defined, retrieve data
        if comp != None:
            if comp not in _comps:
                raise IOError('The specified component does not exist.')
                
            else:                
                self.data = self._read_component_data(self.filename,
                                                      self.header[self.comp]['nbodies'],
                                                      int(self.header[self.comp]['data_start']))
        # wrapup
        self.f.close()
        
    def _read_primary_header(self):
        """read the primary header from an OUT. file"""

        self._check_magic_number()

        # reset to beginning and read current time
        self.f.seek(0)
        self.time, = np.fromfile(self.f, dtype='<f8', count=1)
        self._nbodies_tot, self._ncomp = np.fromfile(self.f, dtype=np.uint32,count=2)

        data_start = 16
        
        for comp in range(0,self._ncomp):
            self.f.seek(data_start)
            next_comp = self._read_out_component_header()
            data_start = next_comp

    def _summarise_primary_header(self):
        """a short summary of what is in the file"""

        ncomponents = len(self.header.keys())
        comp_list   = list(self.header.keys())
        print("Found {} components.".format(ncomponents))

        for n in range(0,ncomponents):
            print("Component {}: {}".format(n,comp_list[n]))
            

    def _read_out_component_header(self):
        """read in the header for a single component, from an OUT. file"""
        
        #_ = f.tell()  # byte position of this component

        if self._float_len == 4:
            _1,_2, nbodies, nint_attr, nfloat_attr, infostringlen = np.fromfile(self.f, dtype=np.uint32, count=6)            
        else:
            nbodies, nint_attr, nfloat_attr, infostringlen = np.fromfile(self.f, dtype=np.uint32, count=4)

        # information string from the header
        head = np.fromfile(self.f, dtype=np.dtype((np.bytes_, infostringlen)),count=1)
        head_normal = head[0].decode()
        head_dict = yaml.safe_load(head_normal)
        
        # deprecated backward compatibility here: see frozen versions if this is an old file

        comp_data_pos = self.f.tell()  # byte position where component data begins

        # the default fields are (m, x, y, z, vx, vy, vz, p)
        nfields = 8
        comp_length = nbodies * (8 * int(head_dict['parameters']['indexing']) +
                                 self._float_len * nfields +
                                 4 * nint_attr +
                                 self._float_len * nfloat_attr)
        comp_data_end = self.f.tell() + comp_length  # byte pos. of comp. data end

        head_dict['nint_attr']   = nint_attr
        head_dict['nfloat_attr'] = nfloat_attr
        head_dict['nbodies']     = nbodies
        head_dict['data_start']  = comp_data_pos
        head_dict['data_end']    = comp_data_end

        self.header[head_dict['name']] = head_dict

        # specifically look for indexing
        try:
            self.indexing = head_dict['parameters']['indexing']
        except:
            self.indexing = head_dict['indexing']=='true'

        return comp_data_end
                
    
    def _check_magic_number(self):
        """check the magic number to see if a file is float or double"""

        self.f.seek(16)  # find magic number
        cmagic, = np.fromfile(self.f, dtype=np.uint32, count=1)

        # check if it is float vs. double
        if cmagic == 2915019716:
            self._float_len = 4
            self._float_str = 'f'
        else:
            self._float_len = 8
            self._float_str = 'd'
        


    def _read_component_data(self,filename,nbodies,offset):
        """read in all data for component"""

        dtype_str = []
        colnames = []
        if self.header[self.comp]['parameters']['indexing']:
            # if indexing is on, the 0th column is Long
            dtype_str = dtype_str + ['l']
            colnames  = colnames + ['index']

        dtype_str = dtype_str + [self._float_str] * 8
        colnames = colnames + ['m', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'potE']

        dtype_str = dtype_str + ['i'] * self.header[self.comp]['nint_attr']
        colnames = colnames + ['i_attr{}'.format(i)
                               for i in range(self.header[self.comp]['nint_attr'])]

        dtype_str = dtype_str + [self._float_str] * self.header[self.comp]['nfloat_attr']
        colnames = colnames + ['f_attr{}'.format(i)
                               for i in range(self.header[self.comp]['nfloat_attr'])]

        dtype = np.dtype(','.join(dtype_str))

        out = np.memmap(filename,
                        dtype=dtype,
                        shape=(1, nbodies),
                        offset=offset,
                        order='F', mode='r')

        tbl = dict()
        for i, name in enumerate(colnames):
            tbl[name] = np.array(out['f{}'.format(i)][0], copy=True)

        del out  # close the memmap instance

        return tbl



