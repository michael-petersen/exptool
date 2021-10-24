"""
reader for the split phase-space protocol (SPL) files from EXP

MSP 28 Sep 2021 Verify initial commit is working

module load python
module load cuda
python3 setup.py install --user



subfiles follow this system:

SPL.  run3.  00062_   0-    1
^     ^      ^        ^     ^
pref  runtag filenum  comp  subfile

"""

import numpy as np

# requires yaml support: likely needs to be installed.
try:
    import yaml
except:
    pass # revert to the backup yaml reader




class Input:
    """Python reader for Split Phase-Space Protocol (SPL) files used by EXP.

    inputs
    ----------
    filename : str
        The SPL filename to load.

    """

    def __init__(self, filename,comp=None, nbodies=-1,verbose=0,legacy=False):
        """
        inputs
        ------------
        filename: str
            filename to open
        comp    : str
            name of the component to return.
        nbodies : integer
            reduce the number of bodies that are returned.
        verbose : integer
            levels of verbosity.
        
        """
        self.verbose = verbose

        # initial check for file validity
        try:
            self.f = open(self.filename, 'rb')
            #self.f.close()
        except Exception:
            raise IOError('Failed to open "{}"'.format(filename))

        if comp == None:
            """if no component specified, print a simple summary of the file contents"""
            self.primary_header = dict()
            self._read_primary_header()
            self._summarise_primary_header()
            return

        self.comp = comp

        # set up assuming the directory of the main file is the same
        # as the subfiles. could add verbose flag to warn?
        self.indir = filename.split('SPL')[0]

        
        self.filename = filename
        self.nbodies = nbodies

        # this has a hardwired max right now...but why would anyone go
        # over this limit?
        self.nbodies = int(np.nanmin([nbodies,1000000000]))
        
        # test for split PSP files
        # TODO

        self.primary_header = dict()
        self.comp_map = dict()
        self.comp_head = dict()
        self.nbodies = 0

        # do an initial read of the header
        self._read_primary_header()

        # now we can query out a specific component
        self._make_file_list(self.comp)

        self.f.close()

        # given the comp, pull the data.
        self._pull_data()

        # make the dictionary version
        if legacy==False:
            self._make_dictionary()

    def _summarise_primary_header(self):
        """a short summary of what is in the file"""

        ncomponents = len(self.comp_head.keys())
        comp_list   = list(self.comp_head.keys())
        print("Found {} components.".format(ncomponents))

        for n in range(0,ncomponents):
            print("Component {}: {}".format(n,comp_list[n]))
            

    def _read_primary_header(self):
        """read the primary header of an SPL file"""

        self.f.seek(0)
        self.time, = np.fromfile(self.f, dtype='<f8', count=1)
        self._nbodies_tot, self._ncomp = np.fromfile(self.f, dtype=np.uint32,count=2)

        # process the subheaders to
        data_start = 16# halo, guaranteed first component location...

        # now read the component headers
        for comp in range(0,self._ncomp):
            self.f.seek(data_start) 
            # manually do headers
            _1,_2,self.nprocs, nbodies, nint_attr, nfloat_attr, infostringlen = np.fromfile(self.f, dtype=np.uint32, count=7)

            # need to figure out what to do with nprocs...it has to always be the same, right?
            head = np.fromfile(self.f, dtype=np.dtype((np.bytes_, infostringlen)),count=1)
            head_normal = head[0].decode()

            try:
                head_dict = yaml.safe_load(head_normal)
            except:
                head_dict = backup_yaml_parse(head)

            #
            head_dict['nint_attr'] = nint_attr
            head_dict['nfloat_attr'] = nfloat_attr
            # data starts at ...
            skip_ahead = 4*7 + infostringlen + data_start
            self.comp_map[head_dict['name']] = skip_ahead
            self.comp_head[head_dict['name']] = head_dict
            try:
                self.indexing = head_dict['parameters']['indexing']
            except:
                self.indexing = head_dict['indexing']=='true'
            data_start = skip_ahead + self.nprocs*1024
    
        
    def _make_file_list(self,comp):
        
        self.f.seek(self.comp_map[comp])
        
        PBUF_SZ = 1024
        PBUF_SM = 32

        self.subfiles = []
    
        for procnum in range(0,self.nprocs):
            PBUF = np.fromfile(self.f, dtype=np.dtype((np.bytes_, PBUF_SZ)),count=1)
            subfile = PBUF[0].split(b'\x00')[0].decode()
            self.subfiles.append(subfile)
            
    def _pull_data(self):
        

        # read in the first one
        tbl = self._read_component_data(self.subfiles[0])

        # make the template based on the first file
        FullParticles = dict()

        # first pass: get everything into memory
        for n in range(0,len(self.subfiles)):
            FullParticles[n] = dict()

            if self.verbose>1:
                print('spl_io._pull_data: On file {} of {}.'.format(n,len(self.subfiles)))
            
            tbl = self._read_component_data(self.subfiles[n])

            for k in tbl.keys():
                FullParticles[n][k] = tbl[k]

        AllParticles = dict()
        for k in tbl.keys():
            AllParticles[k] = np.concatenate([FullParticles[n][k] for n in range(0,len(self.subfiles))])
            
        # now decide on the return format...probably .attribute to
        # match psp_io. but this could change!
        try:
            self.indx = AllParticles['index']
        except:
            pass
        
        self.mass = AllParticles['m']
        self.xpos = AllParticles['x']
        self.ypos = AllParticles['y']
        self.zpos = AllParticles['z']
        self.xvel = AllParticles['vx']
        self.yvel = AllParticles['vy']
        self.zvel = AllParticles['vz']
        self.pote = AllParticles['potE']

        del AllParticles
            
    def _make_dictionary(self):
        """convert the output into a dictionary, in line with psp_io"""

        self.data = dict()

        try:
            self.data['index'] = self.indx
        except:
            pass

        self.data['xpos'] = self.xpos; del self.xpos
        self.data['ypos'] = self.ypos; del self.ypos
        self.data['zpos'] = self.zpos; del self.zpos
        self.data['xvel'] = self.xvel; del self.xvel
        self.data['yvel'] = self.yvel; del self.yvel
        self.data['zvel'] = self.zvel; del self.zvel
        self.data['pote'] = self.pote; del self.pote
        self.data['mass'] = self.mass; del self.mass


    def _handle_subfile(self,subfile):

        _read_component_data(self,subfile)

    def _read_component_data(self,subfile):
        """read in all data for component from individual files

        ,indir,subfile,comp_head
        """
        
        g = open(self.indir+subfile,'rb')
        
        nbodies, = np.fromfile(g, dtype=np.uint32, count=1)
        _float_str = 'f'
        
        dtype_str = []
        colnames = []
        if self.comp_head[self.comp]['parameters']['indexing']:
            # if indexing is on, the 0th column is Long
            dtype_str = dtype_str + ['l']
            colnames = colnames + ['index']
        
        dtype_str = dtype_str + [_float_str] * 8
        colnames = colnames + ['m', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'potE']
        
        dtype_str = dtype_str + ['i'] * self.comp_head[self.comp]['nint_attr']
        colnames = colnames + ['i_attr{}'.format(i)
                               for i in range(self.comp_head[self.comp]['nint_attr'])]
        
        dtype_str = dtype_str + [_float_str] * self.comp_head[self.comp]['nfloat_attr']
        colnames = colnames + ['f_attr{}'.format(i)
                               for i in range(self.comp_head[self.comp]['nfloat_attr'])]
        
        #print(dtype_str)
        dtype = np.dtype(','.join(dtype_str))
        #print(dtype)
        
        out = np.memmap(self.indir+subfile,
                        dtype=dtype,
                        shape=(1, nbodies),
                        offset=4,
                        order='F', mode='r')
        
        tbl = {}
        for i, name in enumerate(colnames):
            if nbodies > 0:
                tbl[name] = np.array(out['f{}'.format(i)][0], copy=True)[0:nbodies]
            else:
                tbl[name] = np.array(out['f{}'.format(i)][0], copy=True)
        
        del out  # close the memmap instance

        g.close() # close the opened file
        
        return tbl





def backup_yaml_parse(yamlin):
    """for absolute compatibility, a very very simple yaml reader
    built to purpose. strips all helpful yaml stuff out, so be careful!"""
    head_dict = {}
    try:
        decoded = yamlin.decode()
    except:
        decoded = yamlin[0].decode()
    split = decoded.split('\n')
    #print(split)
    for k in split:
        #print(k)
        split2 = k.split(':')
        try:
            head_dict[split2[0].lstrip()] = split2[1].lstrip()
        except:
            pass
    return head_dict



