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
MSP 24 Oct 2021 streamline, align with spl_io for merge

PSP is a file format used by the EXP basis function expansion N-body code
written by Martin Weinberg.


TODO
-add protection for missing yaml?
-add handling for multiple components simultaneously
-add _make_dataframe

"""

import numpy as np

# requires yaml support: likely needs to be installed.
import yaml


class Input:
    """Python reader for Phase-Space Protocol (PSP) files used by EXP.

    inputs
    ----------
    filename : str
        The PSP filename to load.

    """

    def __init__(self, filename,comp=None,verbose=0):
        """
        inputs
        ------------
        filename : str
            filename to open
        comp    : str
            name of the component to return.
        verbose : integer
            levels of verbosity.

        
        """
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
        self.comp_head      = dict()
        
        self._read_primary_header()
        
        _comps = list(self.comp_head.keys())
 
        # if a component is defined, retrieve data
        if comp != None:
            if comp not in _comps:
                raise IOError('The specified component does not exist.')
                
            else:
                self.comp = comp
                self.data = self._read_component_data(self.filename,
                                                      self.comp_head[self.comp]['nbodies'],
                                                      int(self.comp_head[self.comp]['data_start']))
            
        # if no header is defined, you will get just the primary header
        
    def _read_primary_header(self):
        """read the primary header"""


        self._check_magic_number()

        # reset to beginning and read current time
        self.f.seek(0)
        self.time, = np.fromfile(self.f, dtype='<f8', count=1)
        self._nbodies_tot, self._ncomp = np.fromfile(self.f, dtype=np.uint32,count=2)

        data_start = 16
        for comp in range(0,self._ncomp):
            self.f.seek(data_start)
            next_comp = self._read_out_component_header(f)
            data_start = next_comp



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
        comp_data_end = f.tell() + comp_length  # byte pos. of comp. data end

        head_dict['nint_attr']   = nint_attr
        head_dict['nfloat_attr'] = nfloat_attr
        head_dict['nbodies']     = nbodies
        head_dict['data_start']  = comp_data_pos
        head_dict['data_end']    = comp_data_end

        self.comp_head[head_dict['name']] = head_dict

        # specifically look for indexing
        try:
            self.indexing = head_dict['parameters']['indexing']
        except:
            self.indexing = head_dict['indexing']=='true'

        return comp_data_end
                
    
    def _check_magic_number(self):

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
        if self.comp_head[self.comp]['parameters']['indexing']:
            # if indexing is on, the 0th column is Long
            dtype_str = dtype_str + ['l']
            colnames  = colnames + ['index']

        dtype_str = dtype_str + [self._float_str] * 8
        colnames = colnames + ['m', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'potE']

        dtype_str = dtype_str + ['i'] * self.comp_head[self.comp]['nint_attr']
        colnames = colnames + ['i_attr{}'.format(i)
                               for i in range(self.comp_head[self.comp]['nint_attr'])]

        dtype_str = dtype_str + [self._float_str] * self.comp_head[self.comp]['nfloat_attr']
        colnames = colnames + ['f_attr{}'.format(i)
                               for i in range(self.comp_head[self.comp]['nfloat_attr'])]

        dtype = np.dtype(','.join(dtype_str))

        out = np.memmap(filename,
                        dtype=dtype,
                        shape=(1, nbodies),
                        offset=offset),
                        order='F', mode='r')

        tbl = dict()
        for i, name in enumerate(colnames):
            tbl[name] = np.array(out['f{}'.format(i)][0], copy=True)

        del out  # close the memmap instance

        return tbl



