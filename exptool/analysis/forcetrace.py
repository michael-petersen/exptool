'''
 _______   ______   .______        ______  _______ .___________..______          ___       ______  _______ 
|   ____| /  __  \  |   _  \      /      ||   ____||           ||   _  \        /   \     /      ||   ____|
|  |__   |  |  |  | |  |_)  |    |  ,----'|  |__   `---|  |----`|  |_)  |      /  ^  \   |  ,----'|  |__   
|   __|  |  |  |  | |      /     |  |     |   __|      |  |     |      /      /  /_\  \  |  |     |   __|  
|  |     |  `--'  | |  |\  \----.|  `----.|  |____     |  |     |  |\  \----./  _____  \ |  `----.|  |____ 
|__|      \______/  | _| `._____| \______||_______|    |__|     | _| `._____/__/     \__\ \______||_______|

forcetrace.py: part of exptool.analysis

7.9.2017: currently a HOWTO sketch on making forcetraces work


# notes on HOWTO
# set up a forcetrace

# 1: print the desired files to a forcetrace-readable file
f = open('/scratch/mpetersen/files.dat','w')

for i in range(900,1001):
    print >>f,'/scratch/mpetersen/Disk001/OUT.run001.%05i'%i

f.close()

#------------------------ bash file to run forcetrace--------------------------------
#!/bin/bash
#SBATCH -t 71:00:00
#SBATCH -o job-%N-%j.out # output file name
#SBATCH -p ib        # queue/partition to submit to
#SBATCH -J forcetrace      # job name
#SBATCH --nodes=4        # number of nodes
#SBATCH --sockets-per-node=2 # attempt to throttle socket number

cd /scratch/mpetersen/Disk001

for (( i = 5; i <= 6; i++ )) do


for (( j = 0; j <= 99; j++ )) do
echo /scratch/mpetersen/Disk001/OUT.run001.$(printf "%05d" $((j+$((i*50))))) >> /scratch/mpetersen/Disk001/file$((i*50)).txt
done



# Let's make some forces!
#   need to copy the desired version of .eof.cache to be .eof.cache.file, then set parameters.

# late time check
mpirun -v forcetrace \
	FILELIST=files.dat \
	OUTFILE=forcetrace001_1000_100k \
	RSCALE=0.0143 \
	NORBMAX=100000 \
	NMAX=36 \
	MMAX=6 \
	NORDER=12 \
	NUMX=128 \
	NUMY=64 >& fprof.out

done

exit 0


# now read back in with python

norb = 50000
nfiles = 50
FT = np.memmap('/scratch/mpetersen/Disk001/forcetrace001_1500',dtype=np.float32,shape=(10,norb,nfiles),mode='r',order='F')




'''
