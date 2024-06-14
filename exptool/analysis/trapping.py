"""
trapping.py (part of exptool)

MSP  8 Dec 2016 Original commit. To be merged with neutrapping.
MSP 23 Dec 2017 Break out bar finding algorithms to the more general pattern.py
MSP  1 Mar 2019 Work on homogenizing docstrings and general commenting
MSP 27 Oct 2021 Enable flexible particle number handling
MSP 18 May 2024 Create HDF5 input/ouput


CLASSES:
ApsFinding
ComputeTrapping (under construction)


TODO:

-Upload example
-Current work is on the kmeans implementations for generalize for all simulations


MAIN REFERENCES:
Petersen, Weinberg, & Katz (2021)
[https://ui.adsabs.harvard.edu/abs/2021MNRAS.500..838P]

Petersen, Weinberg, & Katz (2016)
[https://ui.adsabs.harvard.edu/abs/2016MNRAS.463.1952P]

Calculate quantities for each orbit. Given a set of aps, find:

0. $\langle \theta_{\rm bar}\rangle_N$, the standard trapping metric that assesses the angular separation from the bar for the clusters. Returned value is the maximum angular separation from the bar for the two clusters. $N$ is the number of aps to use in the average.
1. $\langle X_{\rm aps}\rangle_N$, the average position along the bar axis of the aps, by cluster. Then take the minimum of this value. That is, this is the smallest extent of the aps when clustered.
2. $\sigma_{X_{\rm aps}}$, the variance along the bar major axis of the aps positions in a given cluster.
3. Using the ratio of (1) and (2) as a S/N proxy; $x_1$ orbits, as well as higher-order families, have large values. Note that to find $x_2$ orbits, do (1) and (2) in the y-dimension.
4. $\Omega_r$, the $r$ dimension frequency. Used to calculate orbits that fall below the Nyquist frequency for time sampling.

Two improvements over PWK16:
1. Use the closest $N$ aps in time to the indexed time
2. Set a threshold, $T_{\rm thresh}$, that is some multiple of the bar period $T_{\rm bar}$ in which the $N$ aps must reside.

Some combination of these quantities will define the bar.



"""

# general imports
import time
import numpy as np
import datetime
import os
from scipy import interpolate

# io import 
import h5py

# multiprocessing imports
import itertools
from multiprocessing import Pool, freeze_support
import multiprocessing


# exptool imports: relative
from ..io import particle
from ..utils import kmeans
from ..utils import utils
from ..analysis import pattern


class ApsFinding():
    '''
    ApsFinding: a class to find aps

    A standard use would be

    >>> A = trapping.ApsFinding()
    >>> simulation_directory = '/scratch/mpetersen/Disk001/'
    >>> f = open(simulation_directory+'simfiles.dat','w')
    >>> for x in range(000,1000): print(simulation_directory+'OUT.run001.{0:05d}'.format(x))
    >>> f.close()
    >>> trapping_comp = 'star'
    >>> TrappingInstance = A.determine_r_aps(simulation_directory+'simfiles.dat',trapping_comp,particle_indx=np.arange(0,1000,1),out_directory=simulation_directory,return_aps=True)

    To read back in a saved aps file:

    >>> TrappingInstance = A.read_aps_file(aps_file)
    >>> print TrappingInstance['desc']

    Which will make a dictionary of the orbits. The dictionary is _not_ indexed by particle index, but rather from 0 to the number of orbits. The particle index is recorded as TrappingInstance['index'], and may be matched to the dictionary indices.

    '''

    def __init__(self,verbose=0):
        """
        initialise a blank holder for performing the aps finding

        """

        self.verbose = verbose

        return None


    def accept_files(self,filelist,verbose=0):
        """
        parse files from the input list (wrapper)

        """

        self.slist = filelist
        self.verbose = verbose

        ApsFinding.parse_list(self)


    def _parse_list(self):
        """
        parse files from the input list

        the list of files is in a format where each file corresponds to one line.
        full paths are always best, but would work with local paths as well.

        """

        f = open(self.slist)

        s_list = []
        for line in f:
            d = [q for q in line.split()]
            s_list.append(d[0])

        self.SLIST = np.array(s_list)

        if self.verbose >= 1:
            print('exptool.trapping.ApsFinding.parse_list: Accepted {0:d} files.'.format(len(self.SLIST)))


    def determine_r_aps(self,filelist,comp,particle_indx=-1,runtag='',out_directory='',threedee=False,return_aps=False,changingindx=False):
        """
        determine_r_aps

        generate

        """

        # take the inputs and identify all files that we will loop through
        self.slist = filelist
        ApsFinding._parse_list(self)
        # now we have self.SLIST, the parsed list of files we will analyse

        # first, check type of particle_index
        if isinstance(particle_indx,int):
            if particle_indx < 0:
                # if particle_indx < 0, make the comparison index all particles
                Oa = particle.Input(self.SLIST[0],comp=comp,verbose=0)
                particle_indx = np.arange(0,Oa.data['id'].size,1)
            else:
                # limit to the maximum number desired
                particle_indx = np.arange(0,particle_indx,1)

        # assume an array has been passed
        elif isinstance(particle_indx,np.ndarray):
            changingindx = True

        else:
            raise ValueError("exptool.ApsFinding.trapping._determin_r_aps: particle_indx must be an integer or an array.")

        # sort the particle indices
        particle_indx = particle_indx[particle_indx.argsort()]

        # how many orbits are we doing?
        total_orbits = len(particle_indx)

        #
        # stamps the output file with the current time. do we like this?
        #
        tstamp = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d+%H:%M:%S')
        #tstamp = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d+%H')

        # create a new file using the particle number, runtag, and time
        outputfile = out_directory+'RadialAps_N{}_r{}_T{}.h5'.format(total_orbits,runtag,tstamp)
        f = h5py.File(outputfile,"w")

        # createdescriptor string
        desc = 'apsfile for '+comp+' in '+out_directory+', norbits='+str(total_orbits)+', threedee='+str(threedee)+', using '+filelist
        
        # Write the descriptor string as an attribute
        f.attrs['description'] = desc

        # make blank dictionary for the aps
        aps_dictionary = dict() 

        # make a blank array for each orbit
        for i in particle_indx: aps_dictionary[i] = []

        # loop through files
        for i in range(1,len(self.SLIST)-1):

            if i==1:
                # open three files to compare
                Oa = particle.Input(self.SLIST[i-1],comp=comp,verbose=0)
                Ob = particle.Input(self.SLIST[i],comp=comp,verbose=self.verbose)
                Oc = particle.Input(self.SLIST[i+1],comp=comp,verbose=0)
                #Oa = particle.Input(self.SLIST[i-1],legacy=False,comp=comp,verbose=0)
                #Ob = particle.Input(self.SLIST[i],legacy=False,comp=comp,verbose=self.verbose)
                #Oc = particle.Input(self.SLIST[i+1],legacy=False,comp=comp,verbose=0)

                tval = Ob.time
                dt   = Oc.time - Ob.time # dumps must be evenly spaced! (though this is not a requirement later)

                # get the indices correct here
                # create the map from each saved file, based on index
                # should only do this step if index is passed, otherwise it might be expensive
                if changingindx:
                    _1, p1indx, _2  = np.intersect1d(Oa.data['id'], particle_indx, return_indices=True)
                    _1, p2indx, _2  = np.intersect1d(Ob.data['id'], particle_indx, return_indices=True)
                    _1, p3indx, _2  = np.intersect1d(Oc.data['id'], particle_indx, return_indices=True)

                    X1 = Oa.data['x'][p1indx];Y1 = Oa.data['y'][p1indx];Z1 = Oa.data['z'][p1indx];I1 = Oa.data['id'][p1indx]
                    X2 = Ob.data['x'][p2indx];Y2 = Ob.data['y'][p2indx];Z2 = Ob.data['z'][p2indx];I2 = Ob.data['id'][p2indx]
                    X3 = Oc.data['x'][p3indx];Y3 = Oc.data['y'][p3indx];Z3 = Oc.data['z'][p3indx];I3 = Oc.data['id'][p3indx]

                else:
                    X1 = Oa.data['x'];Y1 = Oa.data['y'];Z1 = Oa.data['z'];I1 = Oa.data['id']
                    X2 = Ob.data['x'];Y2 = Ob.data['y'];Z2 = Ob.data['z'];I2 = Ob.data['id']
                    X3 = Oc.data['x'];Y3 = Oc.data['y'];Z3 = Oc.data['z'];I3 = Oc.data['id']


                # compute radial positions
                if threedee:
                    R1 = np.linalg.norm([X1,Y1,Z1],axis=0)
                    R2 = np.linalg.norm([X2,Y2,Z2],axis=0)
                    R3 = np.linalg.norm([X3,Y3,Z3],axis=0)
                else:
                    R1 = np.linalg.norm([X1,Y1],axis=0)
                    R2 = np.linalg.norm([X2,Y2],axis=0)
                    R3 = np.linalg.norm([X3,Y3],axis=0)

            else: # i!=1

                # roll arrays forward
                R1 = R2
                R2 = R3

                I1 = I2
                I2 = I3

                # save the X,Y,Z from the middle file
                X2 = X3
                Y2 = Y3
                Z2 = Z3

                # bring in the new file
                Oc = particle.Input(self.SLIST[i+1],legacy=False,comp=comp,verbose=self.verbose)

                tval = Oc.time - dt

                if changingindx:
                    _1, p3indx, _2 = np.intersect1d(Oc.data['id'], particle_indx, return_indices=True)

                    X3 = Oc.data['x'][p3indx];Y3 = Oc.data['y'][p3indx];Z3 = Oc.data['z'][p3indx];I3 = Oc.data['id'][p3indx]

                else:
                    X3 = Oc.data['x'];Y3 = Oc.data['y'];Z3 = Oc.data['z'];I3 = Oc.data['id']

                if threedee:
                    R3 = np.linalg.norm([X3,Y3,Z3],axis=0)
                else:
                    R3 = np.linalg.norm([X3,Y3],axis=0)


            # R1 might be the shortest, if particles are added, so only compare up to the length of r1
            # indices are always sorted owing to intersect1d, so we can simply truncate
            r1length = len(R1)

            # apocentre condition: there could be more careful checking here
            #      for 'false apocentres'
            #      or for z apocentres, or 3d, etc.
            aps      = np.logical_and( R2[:r1length] > R1, R2[:r1length] > R3[:r1length] )

            # tag orbits with an internal id for looping
            #IN = np.array([i for i in range(0,len(R2))])

            x          = X2[:r1length][aps]
            y          = Y2[:r1length][aps]
            z          = Z2[:r1length][aps]
            id         = I2[:r1length][aps]
            #orderid    = IN[:r1length][aps]

            if self.verbose > 0:
                    print('exptool.ApsFinding.trapping._determine_r_aps: Current time: {0:4.3f}'.format(tval),end='\r', flush=True)

            # the id of the orbit is preserved and used as the dictionary key
            for j in range(0,len(id)):
                aps_dictionary[id[j]].append([tval,x[j],y[j],z[j]])

        # create a tracker for the number of aps per orbit
        self.napsides = np.zeros([total_orbits,2])

        # print a header with the number of orbits
        f.attrs['total_orbits'] = total_orbits

        orbits_with_apocentre = 0

        # go back through all the orbits and write to file
        for j in range(0,total_orbits):

            orbit_aps_array = np.array(aps_dictionary[particle_indx[j]])

            # if there are valid turning points:
            if (len(orbit_aps_array) > 0):

                orbits_with_apocentre += 1

                # count the number of turning points
                naps = len(orbit_aps_array[:,0])  

                # create a dataset with the index of the particle as the tag
                dataset = f.create_dataset(str(particle_indx[j]), data=orbit_aps_array)

                # create attributes for the dataset
                dataset.attrs['naps'] = naps


            # no valid turning points: put in a blank
            else:

                # create a dataset with the index of the particle as the tag
                dataset = f.create_dataset(str(particle_indx[j]), data=np.array([-1.]))

                # create attributes for the dataset
                dataset.attrs['naps'] = 0

                # guard against zero length
                #np.array([1],dtype='i').tofile(f)

                # indices start at 1
                #np.array( np.array(([-1.,-1.,-1.,-1.])).reshape(-1,),dtype='f').tofile(f)


        f.close()

        print('exptool.trapping.ApsFinding.determine_r_aps: found {} orbits (out of {}) with valid apocentres.'.format(orbits_with_apocentre,total_orbits))

        print('exptool.trapping.ApsFinding.determine_r_aps: savefile is {}'.format(outputfile))

        if (return_aps):
            ApsDict = ApsFinding.read_aps_file(self,outputfile)
            return ApsDict


    def read_aps_file(self,aps_file):
        """read in an aps file



        Todo:
            would be great if this looked for the most probable, perhaps by recent date?
        """

        f = open(aps_file,'rb')

        [self.desc] = np.fromfile(f,dtype='S200',count=1)

        [self.norb] = np.fromfile(f,dtype='i',count=1)

        self.aps = dict()

        self.id_array = np.zeros(self.norb,dtype='i')

        for i in range(self.norb):

            [self.id_array[i]] = np.fromfile(f,dtype='i',count=1)

            [naps] = np.fromfile(f,dtype='i',count=1)


            if naps > 0:

                aps_array = np.fromfile(f,dtype='f',count=4*naps)

                self.aps[i] = aps_array.reshape([naps,4])


        f.close()

        ApsDict = ApsFinding.convert_to_dict(self)
        ApsDict['desc']  = self.desc
        ApsDict['norb']  = self.norb
        ApsDict['index'] = np.array(self.id_array)

        return ApsDict


    def convert_to_dict(self):

        # remake Aps File as a dictionary
        ApsDict = dict()

        for i in range(0,self.norb):
            ApsDict[self.id_array[i]] = self.aps[i]

        return ApsDict




class ComputeTrapping:
    '''
    Class to be filled out with the trapping dictionary solver once it is out of prototyping.

    '''
    def __init__(self):

        pass



def write_trapping_file(array,times,filename,tdtype='i1'):
    """write trapping arrays to file

    """
    f = open(filename,'wb')
    np.array([array.shape[0],array.shape[1]],dtype='i').tofile(f)
    np.array(times,dtype='f').tofile(f)
    np.array(array.reshape(-1,),dtype=tdtype).tofile(f)
    f.close()


def read_trapping_file(t_file,tdtype='i1'):
    """read trapping arrays from file

    """
    f = open(t_file,'rb')
    [norb,ntime] = np.fromfile(f,dtype='i',count=2)
    bar_times = np.fromfile(f,dtype='f',count=ntime)
    trap_tmp = np.fromfile(f,dtype=tdtype,count=norb*ntime)
    trap_array = trap_tmp.reshape([norb,ntime])
    return bar_times,trap_array



def reduce_aps_dictionary(TrappingInstance, norb):
    """
    Reduces the apsides data in a trapping instance to only include a specified number of orbits.

    This function takes a trapping instance dictionary and reduces it to include only the
    specified number of orbits (`norb`). It retains the description and the first `norb`
    orbits' data.

    Parameters
    ----------
    TrappingInstance : dict
        A dictionary containing the trapping instance data with multiple orbits.
    norb : int
        The number of orbits to include in the reduced dictionary.

    Returns
    -------
    TrappingInstanceOut : dict
        A reduced dictionary containing only the specified number of orbits
        from the original trapping instance.

    Examples
    --------
    >>> trapping_instance = {
            'desc': 'Sample trapping instance',
            0: {'apside_1': [1, 2], 'apside_2': [3, 4]},
            1: {'apside_1': [5, 6], 'apside_2': [7, 8]},
            2: {'apside_1': [9, 10], 'apside_2': [11, 12]}
        }
    >>> reduced_instance = reduce_aps_dictionary(trapping_instance, 2)
    >>> print(reduced_instance)
    {'norb': 2, 'desc': 'Sample trapping instance', 0: {'apside_1': [1, 2], 'apside_2': [3, 4]}, 1: {'apside_1': [5, 6], 'apside_2': [7, 8]}}
    """
    # Initialize the output dictionary
    TrappingInstanceOut = {}

    # Add the number of orbits to the output dictionary
    TrappingInstanceOut['norb'] = norb

    # Add the description to the output dictionary
    TrappingInstanceOut['desc'] = TrappingInstance['desc']

    # Loop through the specified number of orbits and add them to the output dictionary
    for i in range(norb):
        TrappingInstanceOut[i] = TrappingInstance[i]

    return TrappingInstanceOut


def beane_criteria(K):
    """Implement the criteria from Beane et al. (2024)
    
    works best for polar classifications"""

    # how many clusters?
    k = K.K

    # Compute radii and theta values from clusters
    rad_clusters = np.array([np.linalg.norm(K.clusters[i], axis=1) for i in range(k)])
    the_clusters = np.array([np.arctan2(np.abs(K.clusters[i][:, 1]), np.abs(K.clusters[i][:, 0])) for i in range(k)])

    # implement equation A1: the maximum angle from the bar for the clusters
    thetadiff = np.max([np.arctan2(np.abs(K.mu[i][1]), np.abs(K.mu[i][0])) for i in range(k)])

    # if thetadiff < pi/8, consider the particle trapped

    # implement equation A2:
    clusterstd = np.sum([np.std(rad_clusters[i]) for i in range(k)])
    clustermean = np.sum([np.mean(rad_clusters[i]) for i in range(k)])
    tightness = clusterstd/clustermean

    # if tightness is < 0.22, consider the particle trapped

    return thetadiff,tightness


def evaluate_clusters_polar(K, maxima=False, rank=False, perc=0.):
    """
    Calculate statistics for clusters in polar coordinates.

    This function evaluates the clustering results in polar coordinates (r, theta).
    It computes various statistics such as mean, standard deviation, and optionally
    ranks and percentiles.

    Parameters
    ----------
    K : KMeans
        An instance of a K-means clustering result.
    maxima : bool, optional
        If True, calculate maximum quantities. If False, calculate average quantities.
        Default is False.
    rank : bool, optional
        If True, use rank-ordered statistics. Default is False.
    perc : float, optional
        Percentage threshold for rank ordering. Default is 0.

    Returns
    -------
    theta_n : float
        Angle measure in the context of the clusters.
    clustermean : float
        The mean value of the clusters.
    clusterstd_r : float
        The standard deviation of the clusters in the radial direction.
    clusterstd_t : float
        The standard deviation of the clusters in the angular direction.

    Notes
    -----
    This function computes the radii and theta values from the clusters, then calculates
    either the maximum or average statistics based on the `maxima` parameter. It also
    handles rank-ordered statistics if `rank` is True and `perc` is greater than 0.

    Examples
    --------
    >>> K = kmeans.KMeans(k=2, X=ApsArray)
    >>> K.find_centers()
    >>> theta_n, clustermean, clusterstd_r, clusterstd_t = evaluate_clusters_polar(K)
    """
    # Number of clusters
    k = K.K

    # Check if rank is True but perc is not set
    if rank and perc == 0.:
        raise SyntaxError('exptool.trapping.evaluate_clusters_polar: Perc must be >0.')

    # Compute radii and theta values from clusters
    rad_clusters = np.array([np.linalg.norm(K.clusters[i], axis=1) for i in range(k)])
    the_clusters = np.array([np.arctan2(np.abs(K.clusters[i][:, 1]), np.abs(K.clusters[i][:, 0])) for i in range(k)])

    if maxima:
        # Calculate maximum quantities
        clustermean = np.max([np.mean(rad_clusters[i]) for i in range(k)])
        theta_n = np.max([np.arctan2(np.abs(K.mu[i][1]), np.abs(K.mu[i][0])) for i in range(k)])

        if rank:
            # Use rank-ordered statistics
            organized_rad = np.array([np.sort(rad_clusters[i]) for i in range(k)])
            organized_the = np.array([np.sort(the_clusters[i]) for i in range(k)])
            clusterstd_r = np.max([np.percentile(organized_rad[i] - np.mean(rad_clusters[i]), perc) for i in range(k)])
            clusterstd_t = np.max([np.percentile(organized_the[i] - np.mean(the_clusters[i]), perc) for i in range(k)])
        else:
            clusterstd_r = np.max([np.std(rad_clusters[i]) for i in range(k)])
            clusterstd_t = np.max([np.std(the_clusters[i]) for i in range(k)])
    else:
        # Calculate average quantities
        if rank:
            # Use rank-ordered statistics
            organized_rad = np.array([np.sort(rad_clusters[i]) for i in range(k)])
            organized_the = np.array([np.sort(the_clusters[i]) for i in range(k)])
            clusterstd_r = np.mean([np.percentile(organized_rad[i] - np.mean(rad_clusters[i]), perc) for i in range(k)])
            clusterstd_t = np.mean([np.percentile(organized_the[i] - np.mean(the_clusters[i]), perc) for i in range(k)])
        else:
            clusterstd_r = np.mean([np.std(rad_clusters[i]) for i in range(k)])
            clusterstd_t = np.mean([np.std(the_clusters[i]) for i in range(k)])

        clustermean = np.mean([np.mean(rad_clusters[i]) for i in range(k)])
        theta_n = np.mean([np.arctan2(np.abs(K.mu[i][1]), np.abs(K.mu[i][0])) for i in range(k)])

    return theta_n, clustermean, clusterstd_r, clusterstd_t


def process_kmeans_polar(ApsArray, indx=-1, k=2, maxima=False, rank=False, perc=0.):
    """
    Perform robust K-means clustering on apsidal data in polar coordinates.

    This function performs K-means clustering on the provided apsidal array, 
    computes trapping metrics in polar coordinates, and handles potential 
    edge cases where clusters may have very few points.

    Parameters
    ----------
    ApsArray : array-like
        The array of apsides for an individual orbit. Each element should contain
        the (r, theta) coordinates of an apsis.
    indx : int, optional
        A designation of the orbit, for use with multiprocessing. Default is -1.
    k : int, optional
        The number of clusters to form. Default is 2.
    maxima : bool, optional
        Calculate average (if False) or maximum (if True) quantities. Default is False.
    rank : bool, optional
        Toggle ranking. Default is False.
    perc : float, optional
        Percentage threshold for ranking. Default is 0.

    Returns
    -------
    theta_n : float
        Some angle measure in the context of the clusters.
    clustermean : float
        The mean value of the clusters.
    clusterstd_r : float
        The standard deviation of the clusters in the radial direction.
    clusterstd_theta : float
        The standard deviation of the clusters in the angular direction.
    kmeans_plus_flag : int
        Indicator flag: 0 for successful basic K-means, 1 for successful K-means++,
        2 for failure in both K-means and K-means++.

    Notes
    -----
    This implementation confines the clustering to two dimensions and includes 
    robustness checks to handle small cluster sizes. In case of failure in 
    basic K-means, it retries using the K-means++ initialization method.

    Examples
    --------
    >>> ApsArray = np.array([[1.0, 0.0], [2.0, 1.0], [1.5, 0.5], [3.0, 1.5]])
    >>> theta_n, clustermean, clusterstd_r, clusterstd_theta, flag = process_kmeans_polar(ApsArray)
    """
    kmeans_plus_flag = 0
    K = kmeans.KMeans(k, X=ApsArray)
    K.find_centers()

    # Minimum cluster size threshold
    min_cluster_size = 1

    try:
        clustersize = np.array([np.array(K.clusters[c]).size / 2. for c in range(k)])

        # Ensure no single-point clusters
        while np.min(clustersize) <= min_cluster_size:
            w = np.where(clustersize > min_cluster_size)[0]
            new_aps = np.array([np.concatenate([np.array(K.clusters[x])[:, 0] for x in w]), \
                                np.concatenate([np.array(K.clusters[x])[:, 1] for x in w])]).T

            K = kmeans.KMeans(k, X=new_aps)
            K.find_centers()
            clustersize = np.array([np.array(K.clusters[c]).size / 2. for c in range(k)])

        theta_n, clustermean, clusterstd_r, clusterstd_theta = \
            evaluate_clusters_polar(K, maxima=maxima, rank=rank, perc=perc)

    except:
        # If basic K-means fails, try K-means++
        K = kmeans.KPlusPlus(k, X=ApsArray)
        K.init_centers()
        K.find_centers(method='++')
        kmeans_plus_flag = 1

        try:
            clustersize = np.array([np.array(K.clusters[c]).size / 2. for c in range(k)])

            while np.min(clustersize) <= min_cluster_size:
                w = np.where(clustersize > min_cluster_size)[0]
                new_aps = np.array([np.concatenate([np.array(K.clusters[x])[:, 0] for x in w]), \
                                    np.concatenate([np.array(K.clusters[x])[:, 1] for x in w])]).T

                K = kmeans.KPlusPlus(k, X=new_aps)
                K.init_centers()
                K.find_centers(method='++')
                clustersize = np.array([np.array(K.clusters[c]).size / 2. for c in range(k)])

            theta_n, clustermean, clusterstd_r, clusterstd_theta = \
                evaluate_clusters_polar(K, maxima=maxima, rank=rank, perc=perc)

        except:
            # If both methods fail, set all outputs to NaN
            clusterstd_r = np.nan
            clusterstd_theta = np.nan
            clustermean = np.nan
            theta_n = np.nan
            kmeans_plus_flag = 2

    return theta_n, clustermean, clusterstd_r, clusterstd_theta, kmeans_plus_flag





def process_kmeans(ApsArray,indx=-1,k=2,maxima=False,mad=False):
    '''
    #
    # robust kmeans implementation
    #
    #    -can be edited for speed
    #    -confined to two dimensions

    inputs
    ----------
    ApsArray         : the array of aps for an individual orbit
    indx             : a designation of the orbit, for use with multiprocessing
    k                : the number of clusters
    maxima           : calculate average (if False) or maximum (if True) quantities
    mad              : toggle median absolute deviation calculation


    returns
    ----------
    theta_n          : (see explanation at beginning for definitions)
    clustermean      :
    clusterstd_x     :
    clusterstd_y     :
    kmeans_plus_flag :




    '''
    kmeans_plus_flag = 0
    K = kmeans.KMeans(k,X=ApsArray)
    K.find_centers()

    # find the standard deviation of clusters
    try:

        # these may be better served as maxima
        if ~maxima:

            if ~mad:
                clusterstd_x = np.mean([np.std(np.array(K.clusters[i]),axis=0)[0] for i in range(0,k)])
                clusterstd_y = np.mean([np.std(np.array(K.clusters[i]),axis=0)[1] for i in range(0,k)])

            else:
                # median absolute deviation implementation to check
                clusterstd_x = np.mean([np.median(np.abs(np.array(K.clusters[i]) - np.median(np.array(K.clusters[i]),axis=0)),axis=0)[0]\
                                            for i in range(0,k)])
                clusterstd_y = np.mean([np.median(np.abs(np.array(K.clusters[i]) - np.median(np.array(K.clusters[i]),axis=0)),axis=0)[1]\
                                            for i in range(0,k)])

            clustermean = np.mean([(K.mu[i][0]**2. + K.mu[i][1]**2.)**0.5 for i in range(0,k)])

        else:
            if ~mad:
                clusterstd_x = np.max([np.std(np.array(K.clusters[i]),axis=0)[0] for i in range(0,k)])
                clusterstd_y = np.max([np.std(np.array(K.clusters[i]),axis=0)[1] for i in range(0,k)])

            else:
                clusterstd_x = np.max([np.median(np.abs(np.array(K.clusters[i]) - np.median(np.array(K.clusters[i]),axis=0)),axis=0)[0]\
                                            for i in range(0,k)])
                clusterstd_y = np.max([np.median(np.abs(np.array(K.clusters[i]) - np.median(np.array(K.clusters[i]),axis=0)),axis=0)[1]\
                                            for i in range(0,k)])

            clustermean = np.max([(K.mu[i][0]**2. + K.mu[i][1]**2.)**0.5 for i in range(0,k)])

        # it may also be interesting to record this in a different format. e.g. for orbits that we don't expect to be aligned with the bar exactly
        # also, this is max. could median be useful? or mean?
        theta_n = np.max([abs(np.arctan(K.mu[i][1]/K.mu[i][0])) for i in range(0,k)])

    # failure on basic kmeans
    except:
        K = kmeans.KPlusPlus(2,X=ApsArray)
        K.init_centers()
        K.find_centers(method='++')
        kmeans_plus_flag = 1

        try:

            if ~maxima:

                if ~mad:
                    clusterstd_x = np.mean([np.std(np.array(K.clusters[i]),axis=0)[0] for i in range(0,k)])
                    clusterstd_y = np.mean([np.std(np.array(K.clusters[i]),axis=0)[1] for i in range(0,k)])

                else:
                    # median absolute deviation implementation to check
                    clusterstd_x = np.mean([np.median(np.abs(np.array(K.clusters[i]) - np.median(np.array(K.clusters[i]),axis=0)),axis=0)[0]\
                                            for i in range(0,k)])
                    clusterstd_y = np.mean([np.median(np.abs(np.array(K.clusters[i]) - np.median(np.array(K.clusters[i]),axis=0)),axis=0)[1]\
                                            for i in range(0,k)])

                clustermean = np.mean([(K.mu[i][0]**2. + K.mu[i][1]**2.)**0.5 for i in range(0,k)])

            else:
                if ~mad:
                    clusterstd_x = np.max([np.std(np.array(K.clusters[i]),axis=0)[0] for i in range(0,k)])
                    clusterstd_y = np.max([np.std(np.array(K.clusters[i]),axis=0)[1] for i in range(0,k)])

                else:
                    clusterstd_x = np.max([np.median(np.abs(np.array(K.clusters[i]) - np.median(np.array(K.clusters[i]),axis=0)),axis=0)[0]\
                                            for i in range(0,k)])
                    clusterstd_y = np.max([np.median(np.abs(np.array(K.clusters[i]) - np.median(np.array(K.clusters[i]),axis=0)),axis=0)[1]\
                                            for i in range(0,k)])

                clustermean = np.max([(K.mu[i][0]**2. + K.mu[i][1]**2.)**0.5 for i in range(0,k)])

            theta_n = np.max([abs(np.arctan(K.mu[i][1]/K.mu[i][0])) for i in range(0,k)])

        # failure mode for advanced kmeans
        except:

            #
            # would like a more intelligent way to diagnose
            #if indx >= 0:
            #    print 'Orbit %i even failed in Kmeans++!!' %indx
            clusterstd_x = np.nan
            clusterstd_y = np.nan
            clustermean = np.nan
            theta_n = np.nan
            kmeans_plus_flag = 2



    return theta_n,clustermean,clusterstd_x,clusterstd_y,kmeans_plus_flag





def transform_aps(ApsArray, BarInstance):
    """
    Transform the apsides array into the bar frame of reference.

    This function transforms the apsides array, aligning it with the bar frame as determined
    by the BarInstance. The transformation is offloaded for clarity.

    Parameters
    ----------
    ApsArray : np.ndarray
        The array of apsides, where each row represents a time step and contains
        [time, x_position, y_position].
    BarInstance : object
        An instance that contains information about the bar's position and motion.

    Returns
    -------
    np.ndarray
        The transformed positions in the bar frame. The output array has the same number of rows
        as ApsArray and two columns corresponding to the transformed x and y positions.

    Notes
    -----
    This transformation assumes that the bar motion is in one direction.
    """

    # Find the bar angle positions corresponding to the times in ApsArray
    bar_positions = pattern.find_barangle(ApsArray[:, 0], BarInstance)

    # Initialize the output array for transformed positions
    X = np.zeros([len(ApsArray[:, 1]), 2])

    # Apply the transformation to align with the bar frame
    X[:, 0] = ApsArray[:, 1] * np.cos(bar_positions) - ApsArray[:, 2] * np.sin(bar_positions)
    X[:, 1] = -ApsArray[:, 1] * np.sin(bar_positions) - ApsArray[:, 2] * np.cos(bar_positions)

    return X


def do_single_kmeans_step(TrappingInstanceDict,BarInstance,desired_time,\
                          sbuffer=20,\
                          t_thresh=1.5,\
                          maxima=False,\
                          mad=False,\
                          k=2,\
                          verbose=1,
                              polar=False,\
                              rank=False,perc=0.):
    '''
    do_single_kmeans_step: analyze a desired time in the trapping dictionary


    inputs
    ----------
    TrappingInstanceDict :
    BarInstance          :
    desired_time         :
    sbuffer              : number of closest aps (forward and backward looking) to include in clustering
    t_thresh             :
    maxima               :
    verbose              :
    polar
    rank
    perc

    returns
    ----------
    theta_20
    r_frequency
    x_position
    sigma_x
    sigma_y

    note--all are set to np.nan if unclassifiable for some reason

    '''
    norb = TrappingInstanceDict['norb']

    theta_20 = np.zeros(norb)
    r_frequency = np.zeros(norb)
    x_position = np.zeros(norb)
    sigma_x = np.zeros(norb)
    sigma_y = np.zeros(norb)

    # keep track of base statistics
    skipped_for_aps = 0
    skipped_for_res = 0
    sent_to_kmeans_plus = 0
    failed_kmeans_plus = 0

    t1 = time.time()


    for indx in range(0,norb):
        if ((indx % (norb/100)) == 0) & (verbose > 0):  utils.print_progress(indx,norb,'trapping.do_single_kmeans_step')
        #

        # block loop completely if too few aps
        if len(TrappingInstanceDict[indx][:,0]) < sbuffer:
            skipped_for_aps += 1

            theta_20[indx] = np.nan
            r_frequency[indx] = np.nan
            x_position[indx] = np.nan
            sigma_x[indx] = np.nan
            sigma_y[indx] = np.nan

            continue


        # find the closest aps
        relative_aps_time = np.abs(TrappingInstanceDict[indx][:,0] - desired_time)
        closest_aps = (relative_aps_time).argsort()[0:sbuffer]

        # block loop if furthest aps is above some time threshold
        if relative_aps_time[closest_aps[-1]] > t_thresh:
            skipped_for_res += 1

            theta_20[indx] = np.nan
            r_frequency[indx] = np.nan
            x_position[indx] = np.nan
            sigma_x[indx] = np.nan
            sigma_y[indx] = np.nan

            continue


        # transform to bar frame
        X = transform_aps(TrappingInstanceDict[indx],BarInstance)

        # do k-means

        if polar:
            theta_n,clustermean,clusterstd_x,clusterstd_y,kmeans_plus_flag = process_kmeans_polar(X[closest_aps],k=k,maxima=maxima,rank=rank,perc=perc)

        else:
            theta_n,clustermean,clusterstd_x,clusterstd_y,kmeans_plus_flag = process_kmeans(X[closest_aps],k=k,maxima=maxima,mad=mad)


        if kmeans_plus_flag == 1: sent_to_kmeans_plus += 1

        if kmeans_plus_flag == 2: failed_kmeans_plus += 1

        # set the values for each orbit
        theta_20[indx] = theta_n

        r_frequency[indx] =  1./(TrappingInstanceDict[indx][closest_aps[0],0] -\
                                 TrappingInstanceDict[indx][closest_aps[0]-1,0])

        x_position[indx] = clustermean
        sigma_x[indx] = clusterstd_x
        sigma_y[indx] = clusterstd_y


    if (verbose > 1):
        t2 = time.time()
        print('K-means took {0:3.2f} seconds ({1:3.2f} ms per orbit)'.format(t2, t2/norb*1000))

    print('skipped_for_aps',skipped_for_aps)
    print('skipped_for_res',skipped_for_res)
    print('sent_to_kmeans_plus',sent_to_kmeans_plus)
    print('failed_kmeans_plus',failed_kmeans_plus)


    return theta_20,r_frequency,x_position,sigma_x,sigma_y





def do_kmeans_dict(TrappingInstanceDict,BarInstance,\
                   sbuffer=20,\
                   t_thresh=1.5,\
                   criteria = {},\
                   verbose=0,\
                   maxima=False,\
                   mad=False,\
                       k=2,\
                       polar=False,rank=False,perc=0.):
    '''
    do_kmeans_dict : single processor version of orbit computation

    inputs
    -----------
    TrappingInstanceDict     : the position of apsides for given orbits, dictionary form
    BarInstance              : the bar evolution to use
    sbuffer                  : the number of aps to consider in trapping
    t_thresh                 : the maximum time window to allow trapping (guard against random effects)
    criteria                 : dictionary of families to define (see below)
    verbose                  : reporting flag
    maxima
    mad                      : use median absolute deviation? must be in cartesian
    k                        : how many clusters to compute
    polar                    : calculate cluster statistics in polar coordinates
    rank                     : use rank-ordered?
    perc                     : which percentile to draw for rank-ordered


    returns
    -----------
    trapping_array           : array, [n_families,n_orbits,n_steps]
                                n_families = number of keys in criteria
                                n_orbits   = number of keys in TrappingInstanceDict
                                n_steps    = number of elements in BarInstance.time
                               binary table, with 1 when trapped, 0 when not.


    notes
    -----------
    criteria is a dictionary with the following format:
        criteria[0] = [(0.,0.5),(33.,251.0),(0.0,0.0008),(0.0,0.005),(0.,1.)] # x1 orbits
                 ^        ^         ^             ^           ^         ^
                 |        |         |             |           |         |
                 |     limits on:   |             |           |         |
            family     theta_n    r_freq   sigma-parallel     |    cluster-center
            number                                    sigma-perpendicular




    the Nyquist frequency is calculated based on the minimum spacing of the bar model and is always applied.
       this provides a measure of uncertainty; orbits with unresolved frequencies cannot be classified.


    '''

    norb = TrappingInstanceDict['norb']
    nfamilies = len(criteria.keys())
    if nfamilies == 0:
        return ValueError('exptool.trapping.do_kmeans_dict: no families defined?')


    # set up final array
    trapping_array = np.zeros([nfamilies,norb,len(BarInstance.time)],dtype='i1')
    #
    #
    t1 = time.time()

    # reformat and reconsider me for good printing
    #if (verbose > 0): print 'trapping.do_kmeans_dict: opening angle=%4.3f, OmegaR=%3.2f, sigma_x limit=%4.3f, Aps Buffer=%i' %(opening_angle,rfreq_limit,sigmax_limit,sbuffer)

    for indx in range(0,norb):

        #utils.print_progress(indx,norb,'trapping.do_kmeans_dict')
        if (((indx % (norb/20)) == 0) & (verbose > 0)):
            #
            print(float(indx)/float(norb),'...',end='')
            #utils.print_progress(indx,norb,'trapping.do_kmeans_dict')

        # extract times
        time_sequence = np.array(TrappingInstanceDict[indx])[:,0]

        # guard against total aps range being too small (very important speedup for halo!)
        if time_sequence.size < sbuffer:

            # skip to next orbit
            continue

        # transform to bar frame
        X = transform_aps(TrappingInstanceDict[indx],BarInstance)

        # initialize the list of closest apsides for each timestep with an apse
        orbit_dist = []

        # loop through the apsides
        for midpoint in range(0,len(X)):

            relative_aps_time = time_sequence - time_sequence[midpoint]

            # find the closest aps (not necessarily time-symmetric)
            closest_aps = (abs(relative_aps_time)).argsort()[0:sbuffer]


            # guard against aps with too large of a timespan (some number of bar periods, preferably)
            if relative_aps_time[closest_aps[-1]] > t_thresh:

                # set values to always be 1
                #   would this be better as np.nan?
                #
                orbit_dist.append([0.0,1.0,1.0,1.0,1.0])
                orbit_dist.append([np.max(BarInstance.time),1.0,1.0,1.0,1.0])

                # skip to next orbit
                continue

            # compute the clustering
            if polar:
                theta_n,clustermean,clusterstd_x,clusterstd_y,kmeans_plus_flag = process_kmeans_polar(X[closest_aps],indx,k=k,\
                                                                                                          maxima=maxima,rank=rank,perc=perc)

            else:

                theta_n,clustermean,clusterstd_x,clusterstd_y,kmeans_plus_flag = process_kmeans(X[closest_aps],indx,k=k,maxima=maxima,mad=mad)

            # check time boundaries
            if midpoint==0: # first step

                orbit_dist.append([0.0,theta_n,clusterstd_x,clusterstd_y,clustermean])

            # default action
            orbit_dist.append([time_sequence[midpoint],theta_n,clusterstd_x,clusterstd_y,clustermean])


            if midpoint==(len(X)-1): # last step
                orbit_dist.append([np.max(BarInstance.time),theta_n,clusterstd_x,clusterstd_y,clustermean])


        DD = np.array(orbit_dist) # 0:time, 1:theta_n, 2:sigma_x, 3:sigma_y, 4:<x>

        #nDD = abs(np.ediff1d(DD[:,1],to_begin=1.0))

        # radial frequency computation (spacing between aps!)
        tDD = 1./(abs(np.ediff1d(DD[:,0],to_begin=100.0))+1.e-8)

        # make interpolated functions:
        #     1. theta_n vs time
        #     2. r_frequency vs time
        #     3. sigma_x vs time
        #     4. sigma_y vs time
        theta_func = interpolate.interp1d(DD[:,0],DD[:,1], kind='nearest',fill_value=0.7)      #1
        #
        frequency_func = interpolate.interp1d(DD[:,0],tDD,kind='nearest',fill_value=1.0)       #2
        #
        sigmax_func = interpolate.interp1d(DD[:,0],abs(DD[:,2]),kind='nearest',fill_value=1.0) #3

        sigmay_func = interpolate.interp1d(DD[:,0],abs(DD[:,3]),kind='nearest',fill_value=1.0) #4

        xmean_func = interpolate.interp1d(DD[:,0],abs(DD[:,4]),kind='nearest',fill_value=1.0) #5

        #
        # apply trapping rules
        #

        # set up nyquist frequency limit
        nyquist = 1./(4.*(BarInstance.time[1]-BarInstance.time[0]))


        # hard-code in the metrics that we use. this should be modified to accept various inputs.
        metric = [theta_func(BarInstance.time),\
                  frequency_func(BarInstance.time),\
                 sigmax_func(BarInstance.time),\
                 sigmay_func(BarInstance.time),
                xmean_func(BarInstance.time)]


        for nfam,family in enumerate(np.array(list(criteria.keys()))):

            # how does this get covered from missing criteria?
            #    currently cannot miss criteria
            #
            #    also, how can we add flexibility here?

            trapped = np.where( (metric[0] >= criteria[family][0][0]) & (metric[0] < criteria[family][0][1])\
                      & (metric[1] >= criteria[family][1][0]) & (metric[1] < criteria[family][1][1])\
                      & (metric[2] >= criteria[family][2][0]) & (metric[2] < criteria[family][2][1])\
                      & (metric[3] >= criteria[family][3][0]) & (metric[3] < criteria[family][3][1])\
                      & (metric[4] >= criteria[family][4][0]) & (metric[4] < criteria[family][4][1]))[0]

            trapping_array[nfam,indx,trapped] = np.ones(len(trapped))

    # flag for time elapsed
    t2 = time.time() - t1

    # if extremely verbose, print timing for each orbit
    if (verbose > 1): print('K-means took {0:4.3f} seconds ({0:4.3f} ms per orbit)'.format(t2, t2/norb*1000))

    # wrap as a numpy array and return
    return np.array(trapping_array)


##############################################
#
# MULTIPROCESSING BLOCK
#
#
##############################################



def redistribute_aps(TrappingInstanceDict,divisions,verbose=0):
    '''
    redistribute_aps
       set up aps dictionary to be ready for multiple processors


    inputs
    -------------
    TrappingInstanceDict


    returns
    -------------
    DividedTrappingInstanceDict


    '''

    # initialize holder for particle numbers
    npart = np.zeros(divisions,dtype=object)

    # initialize return structure
    DividedTrappingInstanceDict = [{} for x in range(0,divisions)]

    # compute how many particles each processor is responsible for
    average_part = int(np.floor(TrappingInstanceDict['norb'])/divisions)

    # give leftover particles to first processor
    first_partition = TrappingInstanceDict['norb'] - average_part*(divisions-1)

    if verbose: print('Each processor has {0:d} particles.'.format(average_part))

    low_particle = 0

    # construct the separate processor dictionaries
    for i in range(0,divisions):
        end_particle = low_particle+average_part

        if i==0:
            end_particle = low_particle+first_partition

        for j in range(low_particle,end_particle):
            (DividedTrappingInstanceDict[i])[j-low_particle] = TrappingInstanceDict[j]

        low_particle = end_particle

        if (i>0):
            DividedTrappingInstanceDict[i]['norb'] = average_part
        else:
            DividedTrappingInstanceDict[i]['norb'] = first_partition

    return DividedTrappingInstanceDict




# necessary piece for multiprocessing
def do_kmeans_dict_star(a_b):
    '''Convert `f([1,2])` to `f(1,2)` call.'''
    return do_kmeans_dict(*a_b)



def multi_compute_trapping(DividedTrappingInstanceDict,nprocs,BarInstance,\
                   sbuffer=20,\
                   t_thresh=1.5,\
                   criteria={},\
                   verbose=0,\
                    maxima=False,\
                               mad=False,\
                               k=2,\
                               polar=False,rank=False,perc=0.):
    '''
    multi_compute_trapping
        multiprocessing-enabled kmeans calculator.

    inputs
    ---------------
    DividedTrappingInstanceDict
    nprocs
    BarInstance
    sbuffer
    t_thresh
    criteria
    verbose
    maxima
    mad
    polar
    rank
    perc


    returns
    ---------------
    a_vals                          : trapping arrays broken up by processor



    '''

    pool = Pool(nprocs)
    a_args = [DividedTrappingInstanceDict[i] for i in range(0,nprocs)]
    second_arg = BarInstance
    third_arg = sbuffer
    fourth_arg = t_thresh
    fifth_arg = criteria

    sixth_arg = [0 for i in range(0,nprocs)]
    sixth_arg[0] = verbose

    seventh_arg = maxima
    eighth_arg = mad
    ninth_arg = k
    tenth_arg = polar
    eleventh_arg = rank
    twelvth_arg = perc

    try:
        # this is the python3 version
        a_vals = pool.map(do_kmeans_dict_star, zip(a_args, \
                                                       itertools.repeat(second_arg),\
                                                       itertools.repeat(third_arg),\
                                                       itertools.repeat(fourth_arg),\
                                                       itertools.repeat(fifth_arg),\
                                                       sixth_arg,\
                                                       itertools.repeat(seventh_arg),\
                                                       itertools.repeat(eighth_arg),\
                                                       itertools.repeat(ninth_arg),\
                                                       itertools.repeat(tenth_arg),\
                                                       itertools.repeat(eleventh_arg),\
                                                       itertools.repeat(twelvth_arg)))

    except:
        # this is the python2 version
        a_vals = pool.map(do_kmeans_dict_star, itertools.izip(a_args,\
                                                       itertools.repeat(second_arg),\
                                                       itertools.repeat(third_arg),\
                                                       itertools.repeat(fourth_arg),\
                                                       itertools.repeat(fifth_arg),\
                                                       sixth_arg,\
                                                       itertools.repeat(seventh_arg),\
                                                       itertools.repeat(eighth_arg),\
                                                       itertools.repeat(ninth_arg),\
                                                       itertools.repeat(tenth_arg),\
                                                       itertools.repeat(eleventh_arg),\
                                                       itertools.repeat(twelvth_arg)))

    # clean up to exit
    pool.close()
    pool.join()

    return a_vals



def do_kmeans_multi(TrappingInstanceDict,BarInstance,\
                   sbuffer=20,\
                   t_thresh=1.5,\
                   criteria={},\
                   verbose=0,\
                        maxima=False,\
                        mad = False,\
                        k = 2,\
                        polar=False,rank=False,perc=0.\
                        ):
    '''
    do_kmeans_multi
        multiprocessing-enabled kmeans calculator. Wraps multi_compute_trapping

    inputs
    ---------------
    TrappingInstanceDict  : dictionary, for each orbit a list of aps positions in [t,x_inertial,y_inertial] format
    BarInstance           : pattern.BarInstance, to calculate transformations from aps positions
    sbuffer               : number of apsides to calculate clustering on
    t_thresh              : maximum time window from which apsides should be drawn
    criteria              : dictionary, list of criteria satisfied to make each orbit family
    verbose               : verbosity flag. [what are the levels?]
    maxima
    mad
    k
    polar
    rank
    perc


    returns
    ---------------
    trapped               : dictionary, with an [norbit,ntime] sized array for each input criteria



    '''

    # find number of CPUs accessible
    nprocs = multiprocessing.cpu_count()

    # divide the aps dictionary up for processors
    DividedTrappingInstanceDict = redistribute_aps(TrappingInstanceDict,nprocs)

    if (verbose > 0):
        print('Beginning kmeans, using {0:d} processors.'.format(nprocs))

    t1 = time.time()
    freeze_support()

    # pass specific pieces of dictionary
    trapping_arrays = multi_compute_trapping(DividedTrappingInstanceDict,nprocs,BarInstance,\
                   sbuffer=sbuffer,\
                   t_thresh=t_thresh,\
                   criteria=criteria,\
                   verbose=verbose,\
                   maxima=maxima,\
                   mad=mad,\
                   k=k,\
                   polar=polar,rank=rank,perc=perc)

    print('Total trapping calculation took {0:3.2f} seconds, or {1:3.2f} milliseconds per orbit.'.format(time.time()-t1, 1.e3*(time.time()-t1)/len(TrappingInstanceDict)))

    # go through the dictionary of trapping criteria and re-make the arrays
    trapped = dict()

    for nfam,family in enumerate(np.array(list(criteria.keys()))):

        trapped[nfam] = re_form_trapping_arrays(trapping_arrays,nfam)


    return trapped



def re_form_trapping_arrays(array,array_number):
    '''
    re_form_trapping_arrays
        helper function that undoes the multiprocessing scramble

    inputs
    ----------------
    array           : multidimensional array of orbits
    array_number    : orbit criteria family id number


    returns
    -----------------
    net_array       : singledimensional array of orbits


    '''

    # the arrays are structured as [processor][x1/x2][norb][ntime]

    #  compute the total number of orbits across all processors
    norb_master = 0.0

    for processor in range(0,len(array)): norb_master += array[processor].shape[1]


    #
    # now initialize new blank array?
    #    note that it is initialized as 'i2' to save as much memory as possible
    #
    net_array = np.zeros([int(norb_master),int(array[0].shape[2])],dtype='i2')

    # loop over processors to populate final array
    start_index = 0
    for processor in range(0,len(array)):

        end_index = start_index + array[processor].shape[1]

        net_array[start_index:end_index] = array[processor][array_number]

        start_index = end_index

    return net_array


