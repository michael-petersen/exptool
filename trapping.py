#
# extract global quantities from simulations
#

import time
import numpy as np
import psp_io
import datetime

'''
# try a quick aps-finding exercise


import psp_io
import potential
import helpers
import trapping

B = trapping.BarDetermine()
B = trapping.BarDetermine.read_bar(B,'/scratch/mpetersen/disk007bar.dat')

B = trapping.BarDetermine('/scratch/mpetersen/Disk007/testfiles3.dat',verbose=2)
D = trapping.BarDetermine()
trapping.BarDetermine.accept_inputs(D,'/scratch/mpetersen/Disk004/simlist.dat',verbose=2)

# cut out the duplicate with testlist.dat!!

trapping.BarDetermine.unwrap_bar_position(D)
trapping.BarDetermine.frequency_and_derivative(D,smth_order=0) # go ahead and print the whole thing, can always re-run
trapping.BarDetermine.print_bar(D,'/scratch/mpetersen/disk004bar.dat')


trapping.BarDetermine.frequency_and_derivative(B,smth_order=2)
trapping.BarDetermine.frequency_and_derivative(C,smth_order=2)

plt.figure(0)
plt.plot(B.bar_time,B.bar_deriv,color='gray')
plt.plot(C.bar_time,C.bar_deriv,color='black')
plt.xlabel('Time',size=20)
plt.ylabel('$\\Omega_p$',size=20)
plt.axis([0.4,1.6,0.,100])


import matplotlib.pyplot as plt
import numpy as np

# this whole exercise takes about 30 seconds
Oa = psp_io.Input('/scratch/mpetersen/Disk064a/OUT.run064a.00719',comp='star',verbose=2)
Ob = psp_io.Input('/scratch/mpetersen/Disk064a/OUT.run064a.00720',comp='star',verbose=2)
Oc = psp_io.Input('/scratch/mpetersen/Disk064a/OUT.run064a.00721',comp='star',verbose=2)

Oa.R = (Oa.xpos*Oa.xpos + Oa.ypos*Oa.ypos)**0.5
Ob.R = (Ob.xpos*Ob.xpos + Ob.ypos*Ob.ypos)**0.5
Oc.R = (Oc.xpos*Oc.xpos + Oc.ypos*Oc.ypos)**0.5

aps = np.logical_and(Ob.R>Oa.R,Ob.R>Oc.R)

plt.figure(0)
#plt.scatter(Ob.xpos[aps],Ob.ypos[aps],color='black',s=1.)


yes = np.where( (Ob.R[aps]>0.003) & (Ob.R[aps]<0.01) )[0]

aps_th = np.arctan(Ob.ypos[aps[yes]]/Ob.xpos[aps[yes]])

# bin up the theta positions
binpos = np.linspace(-np.pi/2.,np.pi/2.,60)
hist,bins = np.histogram(aps_th,bins=binpos)

plt.plot(bins[0:-1],hist)
# find the maximum


# could try with phase

yes = np.where( (Ob.R>0.004) & (Ob.R<0.009) )[0]
aval = np.sum(np.cos(2.*np.arctan2(Ob.ypos[yes],Ob.xpos[yes])))
bval = np.sum(np.sin(2.*np.arctan2(Ob.ypos[yes],Ob.xpos[yes])))

print np.arctan2(bval,aval)/2.
# see if this works in time?

f = open('testfiles.dat','w')
for i in range(800,830):
     print '/scratch/mpetersen/Disk007/OUT.run007.%05i'%i
     print >>f,'/scratch/mpetersen/Disk007/OUT.run007.%05i'%i

f.close()

EK = potential.EnergyKappa(Od)


'''

class BarDetermine():

    #
    # class to find the bar
    #

    def __init__(self):
        return None
    
    def accept_inputs(self,filelist,verbose=0):

        self.slist = filelist
        self.verbose = verbose
        
        BarDetermine.cycle_files(self)

    def parse_list(self):
        f = open(self.slist)
        s_list = []
        for line in f:
            d = [q for q in line.split()]
            s_list.append(d[0])

        self.SLIST = np.array(s_list)

        if self.verbose >= 1:
            print 'BarDetermine.parse_list: Accepted %i files.' %len(self.SLIST)

    def determine_r_aps(self,to_file):

        #
        # need to think of the best way to return this data
        #

        if (to_file):
            tstamp = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d+%H:%M:%S')
            f = open('apshold'+tstamp+'.dat','wb+')

        BarDetermine.parse_list(self)

        for i in range(1,len(self.SLIST)-1):

            # open three files to compare
            Oa = psp_io.Input(self.SLIST[i-1],comp='star',verbose=self.verbose)
            Ob = psp_io.Input(self.SLIST[i],comp='star',verbose=self.verbose)
            Oc = psp_io.Input(self.SLIST[i+1],comp='star',verbose=self.verbose)

            # use logic to find aps
            aps = np.logical_and( Ob.R > Oa.R, Ob.R > Oc.R )

            
            indx = np.array([i for i in range(0,len(Ob.xpos))])

            x = Ob.xpos[aps]
            y = Ob.ypos[aps]
            z = Ob.zpos[aps]
            i = indx[aps]

            norb = len(i)
            

            if (to_file):
                np.array( [Ob.ctime],dtype='f').tofile(f)
                q = [0,1,2]#np.array( [i[j],x[j],y[j],z[j]] for j in range(0,norb),dtype='f')
                q.tofile(f)

            #else:
            #    for i in range(0,len
            #    aps.append([

        if (to_file): f.close()

    def cycle_files(self):

        if self.verbose >= 2:
                t1 = time.time()

        BarDetermine.parse_list(self)

        self.bar_time = np.zeros(len(self.SLIST))
        self.bar_pos = np.zeros(len(self.SLIST))

        for i in range(0,len(self.SLIST)):
                O = psp_io.Input(self.SLIST[i],comp='star',verbose=self.verbose)
                self.bar_time[i] = O.ctime
                self.bar_pos[i] = BarDetermine.bar_fourier_compute(self,O.xpos,O.ypos)


        if self.verbose >= 2:
                print 'Computed %i steps in %3.2f minutes, for an average of %3.2f seconds per step.' %( len(self.SLIST),(time.time()-t1)/60.,(time.time()-t1)/float(len(self.SLIST)) )


    def bar_doctor_print(self):

        #
        # wrap the bar file
        #
        BarDetermine.unwrap_bar_position(self)

        BarDetermine.frequency_and_derivative(self)

        BarDetermine.print_bar(self,outfile)

        

    def unwrap_bar_position(self,jbuffer=-1.,smooth=False):

        #
        # modify the bar position to smooth and unwrap
        #
        jnum = 0
        jset = np.zeros_like(self.bar_pos)
        
        for i in range(1,len(self.bar_pos)):
            
            if (self.bar_pos[i]-self.bar_pos[i-1]) < jbuffer:   jnum += 1

            jset[i] = jnum

        self.bar_upos = self.bar_pos + jset*np.pi

        if (smooth):
            self.bar_upos = helpers.savitzky_golay(self.bar_upos,7,3)

        # to unwrap on twopi, simply do:
        #B.bar_upos%(2.*np.pi)

    def frequency_and_derivative(self,smth_order=None,fft_order=None):

        

        if smth_order and fft_order:
            print 'Cannot assure proper functionality of both order smoothing and lo pass filtering.'

        self.bar_deriv = np.zeros_like(self.bar_upos)
        for i in range(1,len(self.bar_upos)):
            self.bar_deriv[i] = (self.bar_upos[i]-self.bar_upos[i-1])/(self.bar_time[i]-self.bar_time[i-1])

            
        if (smth_order):
            smth_params = np.polyfit(self.bar_time, self.bar_deriv, smth_order)
            bar_pos_func = np.poly1d(smth_params)
            self.bar_deriv = bar_pos_func(self.bar_time)

        if (fft_order):
            self.bar_deriv = self.bar_deriv
            
    def bar_fourier_compute(self,posx,posy):

        #
        # use x and y aps positions?
        #
        aval = np.sum( np.cos( 2.*np.arctan2(posy,posx) ) )
        bval = np.sum( np.sin( 2.*np.arctan2(posy,posx) ) )

        return np.arctan2(bval,aval)/2.

    def print_bar(self,outfile):

        #
        # print the barfile.
        #

        f = open(outfile,'w')

        for i in range(0,len(self.bar_time)):
            print >>f,self.bar_time[i],self.bar_upos[i],self.bar_deriv[i]

        f.close()
 
    def place_ellipse(self):

        return None

    def read_bar(self,infile):

        #
        # read a printed bar file
        #

        f = open(infile)

        bar_time = []
        bar_pos = []
        bar_deriv = []
        for line in f:
            q = [float(d) for d in line.split()]
            bar_time.append(q[0])
            bar_pos.append(q[1])
            bar_deriv.append(q[2])

        self.bar_time = np.array(bar_time)
        self.bar_pos = np.array(bar_pos)
        self.bar_deriv = np.array(bar_deriv)

        

class Trapping():

    #
    # class to compute trapping
    #

    def __init__(self):

        return None


    def find_r_aps(self):

        return None
