####################################
#
# Python PSP reader
#
#    MSP 10.25.14
#    Added to exptool 12.3.15
#
import struct
import time
import numpy as np


'''
import psp_io
O = psp_io.Input('/Users/mpetersen/Research/NBody/Disk064a/OUT.run064a.01000',comp='dark',verbose=2)

O = psp_io.Input('/scratch/mpetersen/Disk006/OUT.run006.01000',comp='star',nout=1000)

O = psp_io.Input('/scratch/mpetersen/Disk064a/OUT.run064a.01000',comp='star',orbit_list='/scratch/mpetersen/Disk064a/testolist.dat',infile_list='/scratch/mpetersen/Disk064a/testlist.dat',verbose=1)


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
    #!    orbit_list                                  :    string               :    filename of orbitlist          (ascii, one per line)
    #!    infile_list                                 :    string               :    filename of infilelist         (ascii, one per line)
    
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
    #!     niatr, ndatr,
    #!     return N components, if desired
    #!     do a timing analysis to figure out what is slow, or just generally optimize
    #!           BUT, I suspect it is the array stuffing, which is a clunky for loop

    #! EXAMPLE USAGE
    #! 
    #! import io
    #! O = io.Input(infile)
    #! O.fields              # shows the fields that are available
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


    def __init__(self, infile, comp=None, nout=None, verbose=0, orbit_list=None, infile_list=None):

        
        #
        # master loop
        #

        
        try:
            self.f = open(infile,'rb')
            self.infile = infile
            self.comp = comp
            self.nout = nout
            self.verbose = verbose
            self.orbit_list = orbit_list
            self.infile_list = infile_list

            self.idytpe = 'f4' # internal data type experiment
            
            go_on = 1
            if (self.infile_list): # is this an orbit trace?
                go_on = 2

                
                if not (self.orbit_list):
                    go_on = 0
                    # mode where orbit_list is not defined, but infile is. Read a single orbit
                    #self.singular_orbit = raw_input("Orbit to probe? ")

                if not (self.comp):
                    go_on = 0
                    print 'Component must be defined to proceed with orbit resolution.'

                    
        except:
            print 'The file does not exist.'
            go_on = 0

        
        #
        # if resolving a single timestep
        #
        if go_on == 1:

            #
            # drop into reader routine
            #
            Input.psp_full_read(self)

            # to be expanded and adapted
            self.fields = ['mass','xpos','ypos','zpos','xvel','yvel','zvel','pote']
            
            # clean up your mess!
            self.f.close()

            
        #
        # if resolving orbits across time
        #
        if go_on == 2:

            if self.verbose >= 1:
                print 'Orbit Resolution Initialized...'
            #
            # read a first array to seek_list
            #
            Input.psp_full_read(self,list_make=1)
            self.f.close()

            self.fields = ['MASS','XPOS','YPOS','ZPOS','XVEL','YVEL','ZVEL','POTE','TIME']
            
            #
            # once map is created, drop into orbit retrieval mode
            #
            Input.orbit_resolve(self)

            
    def psp_full_read(self,list_make=None):
        master_time = time.time()

        # read the master header        
        try:
            Input.master_header_read(self)
        except:
            print 'master_header_read() error.'
        
        if self.verbose>=1:
            print 'The time is %3.3f, with %i components and %i total bodies.' %(self.ctime,self.ncomp,self.ntot)

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


        #
        # decide which component, if any, to retrieve
        #
        
        if (self.comp):
            try:
                which_comp = np.where(np.array(self.comp_titles) == self.comp)[0][0]
            except:
                print 'No matching component!'
                which_comp = None
        else:
            which_comp = None

        #
        # if the component is found proceed.
        #
        if (which_comp >= 0):

            #
            # how many bodies to return? (overridden if orbit_list)
            #
            if (self.nout):
                self.return_bodies = self.nout
            else:
                self.return_bodies = self.comp_nbodies[which_comp]
                

            #
            # only return a specified list of orbits?
            #
            
            if (self.orbit_list):
                Input.orbit_map(self,self.orbit_list,which_comp)
                self.return_bodies = len(self.seek_list)
            else:
                self.seek_list = None


            #
            # if returning a single dump, drop into the full component_read loop
            #
            if not (list_make):
                
                if self.verbose >= 1:
                    Input.break_info_string(self,which_comp)


                self.f.seek(self.comp_pos_data[which_comp])
                Input.component_read(self,which_comp)
                 
                if self.verbose >= 2:
                    print 'PSP file read in %3.2f seconds' %(time.time()-master_time)



        

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

        # maybe not necessary, housekeeping
        # convert to private?
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

        # is ndatr the same as the float length, or always double?
        # 8 is the number of fields
        comp_length = self.floatl*nbodies*8 + 4*niatr + self.floatl*ndatr 
        self.comp_data_end[present_comp] = self.f.tell() + comp_length                         # where does the data from this component end?
        
        self.comp_titles[present_comp] = comptitle.strip()
        self.comp_niatr[present_comp] = niatr
        self.comp_ndatr[present_comp] = ndatr
        self.comp_string[present_comp] = str(head)
        self.comp_nbodies[present_comp] = nbodies


    def particle_read(self):
        # to be built when niatr/ndatr is handled
        return None

    def component_read(self,present_comp):

        #
        # cehck for (a) niatr/ndatr, (b) orbit_resolve flag
        #
        if present_comp >= 0:
            if (self.comp_ndatr[present_comp] != 0) | (self.comp_niatr[present_comp] != 0):
                print 'Currently not accepting non-zero attribute fields.'

        '''
        # this would be best as some sort of Particle class
        self.mass = np.zeros(self.return_bodies,dtype=self.idytpe)
        self.xpos = np.zeros(self.return_bodies,dtype=self.idytpe)
        self.ypos = np.zeros(self.return_bodies,dtype=self.idytpe)
        self.zpos = np.zeros(self.return_bodies,dtype=self.idytpe)
        self.xvel = np.zeros(self.return_bodies,dtype=self.idytpe)
        self.yvel = np.zeros(self.return_bodies,dtype=self.idytpe)
        self.zvel = np.zeros(self.return_bodies,dtype=self.idytpe)
        self.pote = np.zeros(self.return_bodies,dtype=self.idytpe)
        '''

        try:
            x = self.seek_list[0]
            seek_flag = 1
        except:
            seek_flag = 0

        if not (seek_flag):

            out = np.memmap(self.infile,dtype=self.dyt,shape=(8,int(self.return_bodies)),offset=int(self.comp_pos_data[present_comp]),order='F')
            self.mass = out[0]
            self.xpos = out[1]
            self.ypos = out[2]
            self.zpos = out[3]
            self.xvel = out[4]
            self.yvel = out[5]
            self.zvel = out[6]
            self.pote = out[7]


        # intelligent scrolling for seek_list considerations?
        if (seek_flag):
            for i in range(0,self.return_bodies):

                # get to position
                self.f.seek(self.seek_list[i])

                # read particle
                [self.mass[i],self.xpos[i],self.ypos[i],self.zpos[i],self.xvel[i],self.yvel[i],self.zvel[i],self.pote[i]] = np.fromfile(self.f, dtype=self.dyt,count=8)


    def break_info_string(self,present_comp):

        #
        # break the info string to be human-readable
        #
        
        head = self.comp_string[present_comp]
        [comptitle,expansion,EJinfo,basisinfo] = [q for q in head.split(':')]

        # -would be awesome to split the info lines and compare to available parameters for a detailed output

        print 'component: ',self.comp_titles[present_comp]
        print 'bodies: ',self.comp_nbodies[present_comp]
        print 'expansion: ',expansion.strip()
        print 'ej info: ',EJinfo
        print 'basis info: ',basisinfo



    def orbit_map(self,orbit_list,present_comp):

        #
        # to make a map of byte positions where the orbits of interest are located
        #
        g = open(orbit_list)
        olist = []
        for line in g:
            d = [q for q in line.split()]
            if len(d)==1: olist.append(float(d[0]))

        g.close()

        OLIST = np.array(olist)

        # override number of bodies to return to match orbit list
        self.return_bodies = len(OLIST)

        self.seek_list = np.zeros(self.return_bodies)
        for i in range(0,self.return_bodies):
            self.seek_list[i] = self.comp_pos_data[present_comp] + OLIST[i]*self.floatl*8. # more for niatr, ndatr



    def orbit_resolve(self):
        res_time_initial = time.time()

        #
        # read the list of infiles
        #
        g = open(self.infile_list)
        ilist = []
        for line in g:
            d = [q for q in line.split()]
            if len(d)==1: ilist.append(d[0])

        g.close()

        ILIST = np.array(ilist)


        #
        # allocate particle arrays
        #
        self.TIME = np.zeros([len(ILIST)])
        self.XPOS = np.zeros([self.return_bodies,len(ILIST)])
        self.YPOS = np.zeros([self.return_bodies,len(ILIST)])
        self.ZPOS = np.zeros([self.return_bodies,len(ILIST)])
        self.XVEL = np.zeros([self.return_bodies,len(ILIST)])
        self.YVEL = np.zeros([self.return_bodies,len(ILIST)])
        self.ZVEL = np.zeros([self.return_bodies,len(ILIST)])
        self.POTE = np.zeros([self.return_bodies,len(ILIST)])

        for i,file in enumerate(ILIST):

            #
            # open the next file
            #
            self.f = open(file,'rb')

            [self.ctime] = np.fromfile(self.f, dtype='<f8',count=1)

            if self.verbose>=4:
                print 'Time: %3.3f' %(self.ctime)

            #
            # read and stuff arrays
            #
            Input.component_read(self,-1)

            # set mass once, which is unchanging (for now!)
            if i==0: self.MASS = self.mass

            
            self.TIME[i] = self.ctime
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

        if self.verbose >= 1:
                    print 'Orbit(s) resolved in %3.2f seconds' %(time.time()-res_time_initial)

            
