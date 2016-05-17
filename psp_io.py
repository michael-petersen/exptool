####################################
#
# Python PSP reader
#
#    MSP 10.25.14
#    Added to exptool 12.3.15
#    niatr/ndatr accepted 3.7.16
#
import time
import numpy as np


'''
#
# various usage examples
#

import psp_io

O = psp_io.Input('/scratch/mpetersen/Disk008/OUT.run008.00430',comp='star',nout=30000)
O = psp_io.Input('/Users/mpetersen/Research/NBody/Disk064a/OUT.run064a.01000',comp='dark',verbose=2)
O = psp_io.Input('/scratch/mpetersen/Disk006/OUT.run006.01000',comp='star',nout=1000)
O = psp_io.Input('/scratch/mpetersen/Disk064a/OUT.run064a.00000',comp='star')
O = psp_io.Input('/scratch/mpetersen/Disk074ashuf/OUT.run074ashuf.00000',comp='star',valid=True)
O = psp_io.Input('/Users/mpetersen/Research/NBody/Disk064a/OUT.run064a.01000',comp='dark',orbit_list='test.dat',infile_list='ilist.dat')


'''

class Input():

    #!
    #! input class to read PSP files
    #!

    #! INPUT fields:
    #!    infile                                      :    string               :    filename to be read
    #!  [optional inputs]
    #!    comp                                        :    string               :    selected component to be read  (default: None)
    #!    nout                                        :    integer              :    how many bodies to return      (default: all)
    #!    verbose                                     :    integer              :    reporting mode, flags below    (default: 0)
    #!    orbit_list                                  :    string               :    filename of orbitlist          (ascii, one integer per line)
    #!    infile_list                                 :    string               :    filename of infilelist         (ascii, one filename per line)
    #!    validate                                    :    boolean              :    validate file and exit         (defalt: False)
    
    #! OUTPUT fields:
    #!    mass,xpos,ypos,zpos,xvel,yvel,zvel,pote     :    float/double         :    particle arrays
    #!       ---OR if multiple timesteps---
    #!    MASS,XPOS,YPOS,ZPOS,XVEL,YVEL,ZVEL,POTE,TIME:    float/double         :    particle arrays [particle, time]

    #! VERBOSITY FLAGS:
    #!  0: Silent
    #!  1: Report Component data
    #!  2: Report Timing data     (in progress)
    #!  4: Report Debug data      (in progress)

    #! WISHLIST:
    #!     
    #!     return N components, if desired
    #!     do a timing analysis to figure out what is slow, or just generally optimize
    #!    

    #! EXAMPLE USAGE
    #! 
    #! import psp_io
    #! O = psp_io.Input(infile)
    #! 
    #!

    # member definitions:
    #
    # __init__
    # psp_full_read
    # master_header_read
    # component_header_read
    # break_info_string
    # component_read
    # orbit_map
    # orbit_resolve


    def __init__(self, infile, comp=None, nout=None, verbose=0, orbit_list=None, infile_list=None, validate=False):

        #
        # set input parameters
        # 
        
        self.infile = infile
        self.comp = comp
        self.nout = nout
        self.verbose = verbose

        # is there an orbit list?
        self.orbit_list = orbit_list
        self.OLIST = None

        # is there an infile list?
        self.infile_list = infile_list
        self.ILIST = None

        #
        # do the components have niatr/ndatr? to be deprecated once I figure out how to write them properly
            
        #
        # set mode based on inputs
        #
        # 0: failure mode
        # 1: read in orbits from single file 
        # 2: read in orbits from multiple files
        # 3: check validity of PSP file

        
        try:
            self.f = open(infile,'rb')

            #
            # default to single file reading
            #
            mode = 1

            #
            # is this an orbit trace?
            #
            if (self.infile_list): 
                mode = 2

                #
                # check specific cases
                #
                if not (self.orbit_list):
                    mode = 0
                    # mode where orbit_list is not defined, but infile is. Read a single orbit
                    #self.singular_orbit = raw_input("Orbit to probe? ")

                if not (self.comp):
                    mode = 0
                    print 'Component must be defined to proceed with orbit resolution.'

            if validate==True:

                mode = 3

                Input.psp_read_headers(self)
        
                if self.verbose>=1:
                    print 'The time is %3.3f, with %i components and %i total bodies.' %(self.ctime,self.ncomp,self.ntot)
            
            
                
                    
        except:
            print 'The master infile is not defined (or does not exist). Master infile required to proceed.'
            mode = 0



        

        
        #
        # single-file mode
        #
        if mode == 1:

            #
            # drop into full reader routine
            #
            Input.psp_full_read(self)

            self.f.close()

            
        #
        # multi-time mode
        #
        if mode == 2:

            if self.verbose >= 1:
                print 'Orbit Resolution Initialized...'
      
            #
            # drop into orbit retrieval mode
            #
            Input.orbit_resolve(self)


        if mode == 0:
            print 'Exiting with error.'
            # would be great to put some error code handling in here


            
    def psp_full_read(self):
        master_time = time.time()

        #
        # do cursory header read
        #
        Input.psp_read_headers(self)
        
        if self.verbose>=1:
            print 'The time is %3.3f, with %i components and %i total bodies.' %(self.ctime,self.ncomp,self.ntot)

        #
        # select component to output
        #
        Input.select_component(self)

        #
        # if the component is found proceed.
        #
        if (self.which_comp >= 0): 

            #
            # how many bodies to return? (overridden if orbit_list)
            #
            if (self.nout):
                self.return_bodies = self.nout
            else:
                self.return_bodies = self.comp_nbodies[self.which_comp]
                

            #
            # only return a specified list of orbits?
            #          
            if (self.orbit_list):
                Input.orbit_map(self)
                self.return_bodies = len(self.OLIST)


            #
            # if returning a single dump, drop into the full component_read loop
            #
            if self.verbose >= 1:
                Input.break_info_string(self)

            #
            # 
            #
            Input.component_read(self)

            
            if self.verbose >= 2:
                print 'PSP file read in %3.2f seconds' %(time.time()-master_time)



        

    def psp_read_headers(self):

        #
        # helper class to read the basic data from multiple headers
        #
        
        # read the master header        
        Input.master_header_read(self)

        #
        # inspect component headers
        #
        present_comp = 0
        while present_comp < self.ncomp:
            
            if self.verbose >= 4:
                print 'Examining component %i' %(present_comp)
                
            # read the component header
            Input.component_header_read(self,present_comp)

            self.f.seek(self.comp_data_end[present_comp])

            present_comp += 1


    def select_component(self):

        #
        # decide which component, if any, to retrieve
        #
        
        if (self.comp):
            try:
                self.which_comp = np.where(np.array(self.comp_titles) == self.comp)[0][0]
            except:
                print 'No matching component!'
                self.which_comp = None
        else:
            self.which_comp = None
            print 'Proceeding without selecting component.'




    def master_header_read(self):
        
        #
        # read the master header
        #
        #    Allocate arrays of necessary component 
        #
        self.f.seek(16) # find magic number
        [cmagic] = np.fromfile(self.f, dtype=np.uint32,count=1)

        # check if it is a float
        # reasonably certain the endianness doesn't affect us here, but verify?
        if cmagic == 2915019716:
            self.floatl = 4
            self.dyt = 'f'
        else:
            self.floatl = 8
            self.dyt = 'd'
            
        # reset to beginning and proceed
        self.f.seek(0)
        [self.ctime] = np.fromfile(self.f, dtype='<f8',count=1)
        [self.ntot,self.ncomp] = np.fromfile(self.f, dtype=np.uint32,count=2)

        self.comp_pos = np.zeros(self.ncomp,dtype=np.uint64)                  # byte position of COMPONENT HEADER for returning easily
        self.comp_pos_data = np.zeros(self.ncomp,dtype=np.uint64)             # byte position of COMPONENT DATA for returning easily
        self.comp_data_end = np.zeros(self.ncomp,dtype=np.uint64)             # byte position of COMPONENT DATA END for returning easily

        # generic PSP items worth making accessible
        self.comp_titles = ['' for i in range(0,self.ncomp)]
        self.comp_niatr = np.zeros(self.ncomp,dtype=np.uint64)                # each component's number of integer attributes
        self.comp_ndatr = np.zeros(self.ncomp,dtype=np.uint64)                # each component's number of double attributes
        self.comp_string = ['' for i in range(0,self.ncomp)]
        self.comp_nbodies = np.zeros(self.ncomp,dtype=np.uint64)              # each component's number of bodies

        
    def component_header_read(self,present_comp):

        self.comp_pos[present_comp] = self.f.tell()
        
        # if PSP changes, this will have to be altered, or I need to figure out a more future-looking version
        if self.floatl==4:
            [cmagic,deadbit,nbodies,niatr,ndatr,infostringlen] = np.fromfile(self.f, dtype=np.uint32,count=6)
        else: 
            [nbodies,niatr,ndatr,infostringlen] = np.fromfile(self.f, dtype=np.uint32,count=4)

        # information string from the header
        head = np.fromfile(self.f, dtype='a'+str(infostringlen),count=1)
        [comptitle,expansion,EJinfo,basisinfo] = [q for q in head[0].split(':')]

        self.comp_pos_data[present_comp] = self.f.tell()            # save where the data actually begins

        # 8 is the number of fields
        comp_length = nbodies*(self.floatl*8 + 4*niatr + self.floatl*ndatr)
        self.comp_data_end[present_comp] = self.f.tell() + comp_length                         # where does the data from this component end?
        
        self.comp_titles[present_comp] = comptitle.strip()
        self.comp_niatr[present_comp] = niatr
        self.comp_ndatr[present_comp] = ndatr
        self.comp_string[present_comp] = str(head)
        self.comp_nbodies[present_comp] = nbodies


    def particle_read(self):
        #
        # routine to return the unique data array based on the particle class
        #

        if self.floatl==4:
            fstring = 'f,f,f,f,f,f,f,f'
            for i in range(0,self.comp_niatr[self.which_comp]): fstring += ',i'
            for i in range(0,self.comp_ndatr[self.which_comp]): fstring += ',f'

        else:
            fstring = 'd,d,d,d,d,d,d,d'
            for i in range(0,self.comp_niatr[self.which_comp]): fstring += ',i'
            for i in range(0,self.comp_ndatr[self.which_comp]): fstring += ',d'

            
        self.readtype = np.dtype(fstring)


    def component_read(self):

        #
        # memmap opens the file itself, so no file open needed here
        #

        #
        # check for (a) niatr/ndatr, (b) orbit_resolve flag
        #
        if (self.which_comp >= 0):
            if (self.comp_ndatr[self.which_comp] != 0) | (self.comp_niatr[self.which_comp] != 0):
                print 'Currently not accepting non-zero attribute fields.'


        #
        # define particle data type
        #      which defines self.readtype
        #

        #Input.particle_read(self)

    

        #
        # gather data field
        #
        if not (self.orbit_list):

            out = np.memmap(self.infile,dtype=self.dyt,shape=(8,int(self.return_bodies)),offset=int(self.comp_pos_data[self.which_comp]),order='F',mode='r')


            self.mass = out[0]
            self.xpos = out[1]
            self.ypos = out[2]
            self.zpos = out[3]
            self.xvel = out[4]
            self.yvel = out[5]
            self.zvel = out[6]
            self.pote = out[7]


        # only looking for a fraction
        if (self.orbit_list):

            # read in all orbits, then obtain specific orbits
            out = np.memmap(self.infile,dtype=self.dyt,shape=(8,int(self.comp_nbodies[self.which_comp])),offset=int(self.comp_pos_data[self.which_comp]),order='F',mode='r')


            self.mass = out[0,self.OLIST]
            self.xpos = out[1,self.OLIST]
            self.ypos = out[2,self.OLIST]
            self.zpos = out[3,self.OLIST]
            self.xvel = out[4,self.OLIST]
            self.yvel = out[5,self.OLIST]
            self.zvel = out[6,self.OLIST]
            self.pote = out[7,self.OLIST]


    def break_info_string(self):

        #
        # break the info string to be human-readable
        #
        
        head = self.comp_string[self.which_comp]
        [comptitle,expansion,EJinfo,basisinfo] = [q for q in head.split(':')]

        print 'component: ',self.comp_titles[self.which_comp]
        print 'bodies: ',self.comp_nbodies[self.which_comp]
        print 'expansion: ',expansion.strip()
        print 'ej info: ',EJinfo
        print 'basis info: ',basisinfo

        # could develop a more user-friendly output for these


    def orbit_map(self):

        #
        # read in the orbit list and convert to an array
        #
        
        g = open(self.orbit_list)
        olist = []
        for line in g:
            d = [q for q in line.split()]
            # no safeguards here yet
            if len(d)==1: olist.append(int(d[0]))

        g.close()

        self.OLIST = np.array(olist)

        # override number of bodies to return to match orbit list
        self.return_bodies = len(self.OLIST)

        if self.verbose >= 1:
            print 'Orbit map accepted with %i bodies.' %self.return_bodies

    def timestep_map(self):

        #
        # read in file list and convert to an array
        #

        g = open(self.infile_list)
        ilist = []
        for line in g:
            d = [q for q in line.split()]
            if len(d)==1: ilist.append(d[0])

        g.close()

        self.ILIST = np.array(ilist)

        if self.verbose >= 1:
            print 'Filename map accepted with %i files (timesteps).' %len(self.ILIST)

        
    def orbit_resolve(self):

        #
        # wrapper to cycle through different files (timesteps) and return orbits
        #
        if self.verbose >= 2:
            res_time_initial = time.time()

        #
        # read a first array to seek_list
        #
        Input.psp_read_headers(self)
        self.f.close()

        if self.verbose>=1:
            print 'The time is %3.3f, with %i components and %i total bodies.' %(self.ctime,self.ncomp,self.ntot)

        #
        # select component to output
        #
        Input.select_component(self)

        #
        # select orbits and files to map
        #
        Input.orbit_map(self)
        Input.timestep_map(self)
        
        #
        # allocate particle arrays
        #
        self.ntimesteps = len(self.ILIST)
        
        self.TIME = np.zeros([self.ntimesteps])
        self.XPOS = np.zeros([self.return_bodies,self.ntimesteps])
        self.YPOS = np.zeros([self.return_bodies,self.ntimesteps])
        self.ZPOS = np.zeros([self.return_bodies,self.ntimesteps])
        self.XVEL = np.zeros([self.return_bodies,self.ntimesteps])
        self.YVEL = np.zeros([self.return_bodies,self.ntimesteps])
        self.ZVEL = np.zeros([self.return_bodies,self.ntimesteps])
        self.POTE = np.zeros([self.return_bodies,self.ntimesteps])

        #
        # cycle through files
        #
        for i,file in enumerate(self.ILIST):

            #
            # open the next file
            #
            self.f = open(file,'rb')

            [ctime] = np.fromfile(self.f, dtype='<f8',count=1)

            if self.verbose>=4:
                print 'Time: %3.3f' %(ctime)

            #
            # read and stuff arrays
            #
            self.infile = file
            Input.component_read(self)

            # set mass once, which is unchanging (for now!)
            if i==0: self.MASS = self.mass

            
            self.TIME[i] = ctime
            #self.MASS[:,i] = self.mass
            self.XPOS[:,i] = self.xpos
            self.YPOS[:,i] = self.ypos
            self.ZPOS[:,i] = self.zpos
            self.XVEL[:,i] = self.xvel
            self.YVEL[:,i] = self.yvel
            self.ZVEL[:,i] = self.zvel
            self.POTE[:,i] = self.pote

            #
            # close file for cleanliness
            #
            self.f.close()

            #
            # delete the individual instances
            #
            del self.mass
            del self.xpos
            del self.ypos
            del self.zpos
            del self.xvel
            del self.yvel
            del self.zvel
            del self.pote

        if self.verbose >= 2:
                    print 'Orbit(s) resolved in %3.2f seconds' %(time.time()-res_time_initial)

            


