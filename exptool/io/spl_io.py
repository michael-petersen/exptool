

import numpy as np

# requires yaml support: likely needs to be installed.
import yaml




class Input:
    """Python reader for Split Phase-Space Protocol (SPL) files used by EXP.

    inputs
    ----------
    filename : str
        The SPL filename to load.

    """

    def __init__(self, filename,comp=None, legacy=True,nbodies=-1,verbose=0,nout=-1):
        """
        inputs
        ------------
        comp    : str
            name of the component to return.
        legacy  : boolean
            if True, adds support for other exptool methods. unneeded if building from scratch.
        nbodies : integer
            reduce the number of bodies that are returned.
        verbose : integer
            levels of verbosity.
        nout    : integer
            deprecated compatibility parameter. use nbodies instead.
        
        """

        
        self.filename = filename
        self.nbodies = nbodies
        
        # deprecated.
        self.infile = filename
        
        # deprecated.
        self.nbodies = int(np.nanmax([nbodies,nout]))
        
        # initial check for file validity
        try:
            f = open(self.filename, 'rb')
            f.close()
        except Exception:
            raise IOError('Failed to load header from file "{}" - are you sure '
                          'this is a PSP file?'.format(filename))

        # test for split PSP files
        # TODO

        # do an initial read of the header
        self.header = self._read_master_header()
        
        _comps = list(self.header.keys())
 

        # if a component is defined, retrieve data
        if comp != None:
            if comp not in _comps:
                raise IOError('The specified component does not exist.')
                
            else:
                self.data = self._read_component_data(self.header[comp])
            
        # if no header is defined, you will get just the master header
        
        if (legacy) & (comp!=None):
            self.header = self.header[comp]
            self.comp = comp
            self._make_backward_compatible()
        elif legacy:
            raise IOError('A component must be specified for legacy usage.')
            

    def _read_component_header(self, f, comp_idx):
        """read in the header for a single component"""
        
        _ = f.tell()  # byte position of this component

        # TODO: if PSP changes, this will have to be altered
        if self._float_len == 4:
            _1,_2,nprocs, nbodies, nint_attr, nfloat_attr, infostringlen = np.fromfile(f, dtype=np.uint32, count=7)
            
        else:
            nprocs, nbodies, nint_attr, nfloat_attr, infostringlen = np.fromfile(f, dtype=np.uint32, count=5)

        # information string from the header
        head = np.fromfile(f, dtype=np.dtype((np.bytes_, infostringlen)),
                           count=1)
        
        # need the backward compatibility here.
        try:
            head_normal = head[0].decode()
            head_dict = yaml.safe_load(head_normal)
        except:
            # backward_compatibility
            head_dict = self._read_compatible_header(head)
            
        comp_data_pos = f.tell()  # byte position where component data begins

        # the default fields are (m, x, y, z, vx, vy, vz, p)
        nfields = 8
        comp_length = nbodies * (8 * int(head_dict['parameters']['indexing']) +
                                 self._float_len * nfields +
                                 4 * nint_attr +
                                 self._float_len * nfloat_attr)
        comp_data_end = f.tell() + comp_length  # byte pos. of comp. data end

        data = dict()
        data['index'] = comp_idx
        for k in head_dict:
            data[k] = head_dict[k]
        data['nint_attr'] = nint_attr
        data['nfloat_attr'] = nfloat_attr
        data['nbodies'] = nbodies
        data['data_start'] = comp_data_pos
        data['data_end'] = comp_data_end
        f.seek(comp_data_end)

        return data
    
    def _read_compatible_header(self,head):
        """read the old style of PSP header
        
        handling could be more general: this may have failure cases that I have not foreseen.
        
        """
        
        head_sep = head[0].decode().split(':')
        head_dict = dict()
        head_dict['parameters'] = dict()
        head_dict['parameters']['indexing'] = 0
        
        for istanza,stanza in enumerate(head_sep):
            
            if istanza==0:
                head_dict['name'] = stanza.strip()
                
            if istanza==1:
                head_dict['id'] = stanza.strip()
                
            if istanza > 1:
                stanza_sep = stanza.split(',')
                for param in stanza_sep:
                    head_dict['parameters'][param.split('=')[0].strip()] = param.split('=')[1].strip()

        return head_dict

    def _read_master_header(self):
        """read the master header of the SPL file"""

        master_header = dict()
        nbodies = 0

        with open(self.filename, 'rb') as f:

            f.seek(16)  # find magic number
            cmagic, = np.fromfile(f, dtype=np.uint32, count=1)

            # check if it is float vs. double
            if cmagic == 2915019716:
                self._float_len = 4
                self._float_str = 'f'
            else:
                self._float_len = 8
                self._float_str = 'd'

            # reset to beginning and read current time
            f.seek(0)
            self.time, = np.fromfile(f, dtype='<f8', count=1)
            self._nbodies_tot, self._ncomp = np.fromfile(f, dtype=np.uint32,
                                                         count=2)

            for i in range(self._ncomp):
                data = self._read_component_header(f, i)
                master_header[data.pop('name')] = data
                nbodies += data['nbodies']

        master_header['nbodies'] = nbodies

        return master_header

    def _read_component_data(self, comp_header):
        """read in all data for component"""

        dtype_str = []
        colnames = []
        if comp_header['parameters']['indexing']:
            # if indexing is on, the 0th column is Long
            dtype_str = dtype_str + ['l']
            colnames = colnames + ['index']

        dtype_str = dtype_str + [self._float_str] * 8
        colnames = colnames + ['m', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'potE']

        dtype_str = dtype_str + ['i'] * comp_header['nint_attr']
        colnames = colnames + ['i_attr{}'.format(i)
                               for i in range(comp_header['nint_attr'])]

        dtype_str = dtype_str + [self._float_str] * comp_header['nfloat_attr']
        colnames = colnames + ['f_attr{}'.format(i)
                               for i in range(comp_header['nfloat_attr'])]

        dtype = np.dtype(','.join(dtype_str))

        out = np.memmap(self.filename,
                        dtype=dtype,
                        shape=(1, comp_header['nbodies']),
                        offset=int(comp_header['data_start']),
                        order='F', mode='r')

        tbl = {}
        for i, name in enumerate(colnames):
            if self.nbodies > 0:
                tbl[name] = np.array(out['f{}'.format(i)][0], copy=True)[0:self.nbodies]
            else:
                tbl[name] = np.array(out['f{}'.format(i)][0], copy=True)

        del out  # close the memmap instance

        return tbl


