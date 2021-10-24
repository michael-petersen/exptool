"""
reader for the split phase-space protocol (SPL) files from EXP

MSP 28 Sep 2021 Verify initial commit is working
MSP 24 Oct 2021 Cleanup to match psp_io



subfiles follow this system:

SPL.  run3.  00062_   0-    1
^     ^      ^        ^     ^
pref  runtag filenum  comp  subfile

"""

import numpy as np

try:
    # requires yaml support: likely needs to be installed.
    import yaml
except ImportError:
    raise ImportError("You will need to 'pip install pyyaml' to use this reader.")
    

class Input:
    """Python reader for Split Phase-Space Protocol (SPL) files used by EXP.

    inputs
    ----------
    filename : str
        The SPL filename to load.

    """

    def __init__(self, filename,comp=None,verbose=0):
        """
        inputs
        ------------
        filename: str
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

        if comp == None:
            self._summarise_primary_header()
            return

        self.comp = comp

        # set up assuming the directory of the main file is the same
        # as the subfiles. could add verbose flag to warn?
        self.indir = filename.split('SPL')[0]

        # now we can query out a specific component
        self._make_spl_file_list(self.comp)

        self.f.close()

        # given the comp, pull the data.
        self._read_spl_component_data()


    def _read_primary_header(self):
        """read the primary header"""

        self._check_magic_number()
        
        self.f.seek(0)
        self.time, = np.fromfile(self.f, dtype='<f8', count=1)
        self._nbodies_tot, self._ncomp = np.fromfile(self.f, dtype=np.uint32,count=2)

        data_start = 16 # guaranteed first component location...

        # now read the component headers
        for comp in range(0,self._ncomp):
            self.f.seek(data_start) 
            next_comp = self._read_spl_component_header()               
            data_start = next_comp 
    

    def _read_spl_component_header(self):
        """read in the header for a single component, from an SPL. file"""

        data_start = self.f.tell()
        # manually do headers
        _1,_2,self.nprocs, nbodies, nint_attr, nfloat_attr, infostringlen = np.fromfile(self.f, dtype=np.uint32, count=7)

        # need to figure out what to do with nprocs...it has to always be the same, right?
        head = np.fromfile(self.f, dtype=np.dtype((np.bytes_, infostringlen)),count=1)
        head_normal = head[0].decode()
        head_dict = yaml.safe_load(head_normal)

        
        head_dict['nint_attr']   = nint_attr
        head_dict['nfloat_attr'] = nfloat_attr
        # data starts at ...
        next_comp = 4*7 + infostringlen + data_start
            
        self.comp_map[head_dict['name']]  = next_comp
        self.comp_head[head_dict['name']] = head_dict

        next_comp +=  self.nprocs*1024

        # specifically look for indexing
        try:
            self.indexing = head_dict['parameters']['indexing']
        except:
            self.indexing = head_dict['indexing']=='true'
        
        return next_comp

                
    def _make_spl_file_list(self,comp):
        
        self.f.seek(self.comp_map[comp])
        
        PBUF_SZ = 1024
        PBUF_SM = 32

        self.subfiles = []
    
        for procnum in range(0,self.nprocs):
            PBUF = np.fromfile(self.f, dtype=np.dtype((np.bytes_, PBUF_SZ)),count=1)
            subfile = PBUF[0].split(b'\x00')[0].decode()
            self.subfiles.append(subfile)

    def _summarise_primary_header(self):
        """a short summary of what is in the file"""

        ncomponents = len(self.comp_head.keys())
        comp_list   = list(self.comp_head.keys())
        print("Found {} components.".format(ncomponents))

        for n in range(0,ncomponents):
            print("Component {}: {}".format(n,comp_list[n]))
            

    def _read_spl_component_data(self):
        
        # read in the first one
        #tbl = self._read_component_data(self.subfiles[0])

        # make the template based on the first file
        FullParticles = dict()

        # first pass: get everything into memory
        for n in range(0,len(self.subfiles)):
            FullParticles[n] = dict()

            if self.verbose>1:
                print('spl_io._read_spl_component_data: On file {} of {}.'.format(n,len(self.subfiles)))
            
            tbl = self._handle_spl_subfile(self.subfiles[n])

            for k in tbl.keys():
                FullParticles[n][k] = tbl[k]

        # construct a single dictionary for the particles
        self.data = dict()
        for k in tbl.keys():
            self.data[k] = np.concatenate([FullParticles[n][k] for n in range(0,len(self.subfiles))])

        # cleanup...
        del FullParticles
        
            
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
        

    def _handle_spl_subfile(self,subfilename):

        subfile = open(self.indir+subfilename,'rb')
        nbodies, = np.fromfile(subfile, dtype=np.uint32, count=1)
        subfile.close() # close the opened file

        # read the data
        tbl = self._read_component_data(self.indir+subfilename,nbodies,4)
        # the offset is always fixed in SPL subfiles

        return tbl


    def _read_component_data(self,filename,nbodies,offset):
        """read in all data for component"""
        
        dtype_str = []
        colnames  = []
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
                        offset=offset,
                        order='F', mode='r')
        
        tbl = dict()
        for i, name in enumerate(colnames):
            tbl[name] = np.array(out['f{}'.format(i)][0], copy=True)
        
        del out  # close the memmap instance
        
        return tbl




