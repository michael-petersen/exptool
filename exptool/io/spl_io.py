"""
reader for the split phase-space protocol (SPL) files from EXP





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

    def __init__(self, filename,comp=None, nbodies=-1,verbose=0):
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

        # this has a hardwired max right now...but why would anyone go
        # over this limit?
        self.nbodies = int(np.nanmin([nbodies,1000000000]))
        
        # initial check for file validity
        try:
            f = open(self.filename, 'rb')
            f.close()
        except Exception:
            raise IOError('Failed to open "{}"'.format(filename))

        # test for split PSP files
        # TODO

        # do an initial read of the header
        self._read_primary_header()
        


    def _read_primary_header(self):

        self.f.seek(0)
        self.time, = np.fromfile(f, dtype='<f8', count=1)
        self._nbodies_tot, self._ncomp = np.fromfile(f, dtype=np.uint32,count=2)
        #print(time,_nbodies_tot,_ncomp)

        # process the subheaders to




        
    def _make_file_list(f,comp_map,comp):
        
        f.seek(comp_map[comp])
        
        PBUF_SZ = 1024
        PBUF_SM = 32
    
        for procnum in range(0,nprocs):
            PBUF = np.fromfile(f, dtype=np.dtype((np.bytes_, PBUF_SZ)),count=1)
            subfile = PBUF[0].split(b'\x00')[0].decode()
            #print(subfile)
            # and then in here, also read the individual files
            #tbl = _read_component_data(indir,subfile,head_dict)
            #print(tbl.keys())
    

            #_make_file_list(f,comp_map,'bulge')






def _read_component_data(indir,subfile,comp_header):
        """read in all data for component"""
        
        g = open(indir+subfile,'rb')
        
        nbodies, = np.fromfile(g, dtype=np.uint32, count=1)
        _float_str = 'f'
        
        dtype_str = []
        colnames = []
        if comp_header['parameters']['indexing']:
            # if indexing is on, the 0th column is Long
            dtype_str = dtype_str + ['l']
            colnames = colnames + ['index']
        
        dtype_str = dtype_str + [_float_str] * 8
        colnames = colnames + ['m', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'potE']
        
        dtype_str = dtype_str + ['i'] * comp_header['nint_attr']
        colnames = colnames + ['i_attr{}'.format(i)
                               for i in range(comp_header['nint_attr'])]
        
        dtype_str = dtype_str + [_float_str] * comp_header['nfloat_attr']
        colnames = colnames + ['f_attr{}'.format(i)
                               for i in range(comp_header['nfloat_attr'])]
        
        print(dtype_str)
        dtype = np.dtype(','.join(dtype_str))
        print(dtype)
        
        out = np.memmap(indir+subfile,
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





"""
indir = '/proj/weinberg/Nbody/Elena/'
filename = '/proj/weinberg/Nbody/Elena/SPL.run3.00315'

filename = '/nas/astro-th/weinberg/Nbody/Elena/Run3/SPL.run3.00062'





def backup_yaml_parse(yamlin):
    head_dict = dict()
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




f = open(filename,'rb')



master_header = dict()
comp_map = dict()
comp_head = dict()
nbodies = 0

f.seek(0)
time, = np.fromfile(f, dtype='<f8', count=1)
_nbodies_tot, _ncomp = np.fromfile(f, dtype=np.uint32,count=2)
print(time,_nbodies_tot,_ncomp)


data_start = 16# halo, guaranteed first component loation...
for comp in range(0,_ncomp):
    f.seek(data_start) 
    # manually do headers
    _1,_2,nprocs, nbodies, nint_attr, nfloat_attr, infostringlen = np.fromfile(f, dtype=np.uint32, count=7)
    # need to figure out what to do with nprocs...
    head = np.fromfile(f, dtype=np.dtype((np.bytes_, infostringlen)),count=1)
    head_normal = head[0].decode()
    try:
        head_dict = yaml.safe_load(head_normal)
    except:
        head_dict = backup_yaml_parse(head)
    #
    head_dict['nint_attr'] = nint_attr
    head_dict['nfloat_attr'] = nfloat_attr
    # data starts at 
    skip_ahead = 4*7 + infostringlen + data_start
    comp_map[head_dict['name']] = skip_ahead
    comp_head[head_dict['name']] = head_dict
    try:
        indexing = head_dict['parameters']['indexing']
    except:
        indexing = head_dict['indexing']=='true'
    data_start = skip_ahead + nprocs*1024
    

    
print(comp_map)
# now comp_map has the starting place for each file



#f.seek(197676) # bulge
#f.seek(395336) # for stars

def _make_file_list(f,comp_map,comp,nprocs):
    #
    f.seek(comp_map[comp])
    PBUF_SZ = 1024
    PBUF_SM = 32
    #
    for procnum in range(0,nprocs):
        #PBUF = np.fromfile(f, dtype=np.dtype((np.bytes_, PBUF_SM)),count=PBUF_SZ/PBUF_SM)
        PBUF = np.fromfile(f, dtype=np.dtype((np.bytes_, PBUF_SZ)),count=1)
        #print(PBUF[0])
        subfile = PBUF[0].split(b'\x00')[0].decode()
        print(subfile)
        # and then in here, also read the individual files
        #tbl = _read_component_data(indir,subfile,head_dict)
        #print(tbl.keys())
    

_make_file_list(f,comp_map,'bulge')




f.seek(1068) # for dark
f.seek(198728) # for bulge
f.seek(396388) # for stars
PBUF_SZ = 1024
PBUF_SM = 32

for procnum in range(0,1):#nprocs):
    #PBUF = np.fromfile(f, dtype=np.dtype((np.bytes_, PBUF_SM)),count=PBUF_SZ/PBUF_SM)
    PBUF = np.fromfile(f, dtype=np.dtype((np.bytes_, PBUF_SZ)),count=1)
    #print(PBUF[0])
    subfile = PBUF[0].split('\x00')[0].decode()
    print(subfile)
    # and then in here, also read the individual files
    tbl = _read_component_data(indir,subfile,head_dict)
    print(tbl.keys())
    

    #g = open(indir+subfile,'rb')
    #nbodfile, = np.fromfile(g, dtype=np.uint32, count=1)
    #print(nbodfile)



"""
