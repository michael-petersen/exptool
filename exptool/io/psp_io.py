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

def _to_python(obj):
    """
    Recursively convert numpy types to native Python types.

    This helper function ensures that objects containing numpy types (such as numpy scalars or arrays)
    are converted to their native Python equivalents. This is particularly important for YAML serialization,
    which may not handle numpy types correctly.

    Parameters
    ----------
    obj : any
        Any Python object, potentially containing numpy types (e.g., numpy scalars, arrays, or nested structures).

    Returns
    -------
    out : any
        The input object with all numpy types converted to native Python types.
    """
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_python(v) for v in obj]
    elif hasattr(obj, 'item'):   # catches numpy scalars
        return obj.item()
    else:
        return obj


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
        id   : int, the integer index of the particle
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

        # initialise dictionary for headers
        self.header      = dict()

        self._read_primary_header()

        self.comp = comp
        _comps = list(self.header.keys())


        # if a component is defined, retrieve data
        if comp != None:

            # or check if we are reading all components
            if comp == 'all':
                self.data = dict()
                for c in _comps:
                    self.data[c] = self._read_component_data(self.filename,
                                                             c,
                                                             self.header[c]['nbodies'],
                                                             int(self.header[c]['data_start']))

            else:
                if comp not in _comps:
                    raise IOError(f'The specified component, {comp}, does not exist.')
                
                else:
                    self.data = self._read_component_data(self.filename,
                                                        self.comp,
                                                        self.header[self.comp]['nbodies'],
                                                        int(self.header[self.comp]['data_start']))
                    
        # wrapup
        self.f.close()

    def write(self, filename):
        """
        Write the current data to a PSP/OUT. file.

        Parameters
        ----------
        filename : str
            The output filename to which the data will be written.

        Behavior
        --------
        Writes all components to the specified file if `comp='all'` was used when reading.
        Writing of single components is not implemented and will raise an exception.

        Exceptions
        ----------
        NotImplementedError
            Raised if attempting to write when `comp` is not 'all'.

        Example
        -------
        >>> inp = Input("input.OUT", comp="all")
        >>> inp.write("output.OUT")
        """
        if self.comp != 'all':
            raise NotImplementedError("Writing single components is not implemented yet. Use comp='all'.")
        
        with open(filename, 'wb') as f:
            self._write_primary_header(f)

            # Now write all component headers and data sequentially.
            for comp in self.header:
                self._write_component_header(f, self.header[comp])
                self._write_component_data(f, comp, self.data[comp])


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

    def _write_primary_header(self, f):
        # time is always <f8
        np.array([self.time], dtype='<f8').tofile(f)

        # total nbodies and number of components
        total = sum(self.header[c]['nbodies'] for c in self.header)
        ncomp = len(self.header)
        np.array([total, ncomp], dtype=np.uint32).tofile(f)

        # next the component headers must follow immediately
        # magic number goes at byte 16
        # TODO: Writing the magic number is required for full PSP file compatibility,
        # but this is not yet implemented. Uncomment and verify the following lines
        # when ready to support the magic number in the output file format.
        #f.seek(16)
        #magic = 2915019716 if self._float_len == 4 else 0
        #np.array([magic], dtype=np.uint32).tofile(f)
        # double up the magic number for consistency
        #np.array([magic], dtype=np.uint32).tofile(f)
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
        # https://raw.githubusercontent.com/michael-petersen/exptool/f5de2b380dd73e31ab8015d366ac44b0b41a2e18/exptool/io/psp_io.py

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
        head_dict['info_len']     = infostringlen
        head_dict['info_str']     = head_normal

        self.header[head_dict['name']] = head_dict

        # specifically look for indexing
        try:
            self.indexing = head_dict['parameters']['indexing']
        except:
            self.indexing = head_dict['indexing']=='true'
            head_dict['parameters']['indexing'] = self.indexing

        return comp_data_end

    def _write_component_header(self, f, compdict):
        """
        compdict is one of self.header[name]
        """

        nbodies     = _to_python(compdict['nbodies'])
        nint_attr   = _to_python(compdict['nint_attr'])
        nfloat_attr = _to_python(compdict['nfloat_attr'])

        compdict_clean = _to_python(compdict)

        info        = yaml.safe_dump(compdict_clean)
        info_bytes = info.encode()
        info_len   = len(info_bytes)

        if self._float_len == 4:
            # the 6 numbers at the start: magic number + 5 (so only write 5)
            magic = 2915019716
            arr = np.array([magic, 0, nbodies,
                            nint_attr, nfloat_attr, info_len],
                            dtype=np.uint32)
        else:
            arr = np.array([nbodies, nint_attr,
                            nfloat_attr, info_len],
                            dtype=np.uint32)

        arr.tofile(f)
        f.write(info_bytes)


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



    def _read_component_data(self,filename,comp,nbodies,offset):
        """read in all data for component"""

        dtype_str = []
        colnames = []
        if self.header[comp]['parameters']['indexing']:
            # if indexing is on, the 0th column is Long
            dtype_str = dtype_str + ['l']
            colnames  = colnames + ['id']

        dtype_str = dtype_str + [self._float_str] * 8
        colnames = colnames + ['m', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'potE']

        dtype_str = dtype_str + ['i'] * self.header[comp]['nint_attr']
        colnames = colnames + ['i_attr{}'.format(i)
                               for i in range(self.header[comp]['nint_attr'])]

        dtype_str = dtype_str + [self._float_str] * self.header[comp]['nfloat_attr']
        colnames = colnames + ['f_attr{}'.format(i)
                               for i in range(self.header[comp]['nfloat_attr'])]

        dtype = np.dtype(','.join(dtype_str))

        # a typical dtype with indexing on will look like:
        # dtype([('f0', '<i8'), ('f1', '<f8'), ('f2', '<f8'), ('f3', '<f8'), ('f4', '<f8'), ('f5', '<f8'), ('f6', '<f8'), ('f7', '<f8'), ('f8', '<f8')])

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

    def _write_component_data(self, f, comp, data):
        compdict = self.header[comp]

        dtype_str = []
        colnames  = []

        if compdict['parameters']['indexing']:
            dtype_str.append('l')
            colnames.append('id')

        dtype_str += [self._float_str] * 8
        colnames  += ['m','x','y','z','vx','vy','vz','potE']

        dtype_str += ['i'] * compdict['nint_attr']
        colnames  += [f'i_attr{i}' for i in range(compdict['nint_attr'])]

        dtype_str += [self._float_str] * compdict['nfloat_attr']
        colnames  += [f'f_attr{i}' for i in range(compdict['nfloat_attr'])]

        dtype = np.dtype(','.join(dtype_str))

        outarr = np.zeros(compdict['nbodies'], dtype=dtype)
        for i, key in enumerate(colnames):
            outarr[f'f{i}'] = data[key]

        outarr.tofile(f)
