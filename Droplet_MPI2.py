import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py as h5
import sys
import os
from pathlib import Path

import psana as ps
import ImgAlgos.PyAlgos
import droplet_module_v3 as dr
import scipy.ndimage as ndimage
import time

# MPI for parallelization
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numWorkers = comm.Get_size()
assert numWorkers>1, 'At least 2 MPI ranks required'


# Instantiate the parser
import argparse
parser = argparse.ArgumentParser(description='Run psana on selected run')
parser.add_argument('-e','--exp', type=str, help='Experiment name')
parser.add_argument('-r','--run', type=int, help='Run to be analyzed')
parser.add_argument('-f','--file', type=str, help='Output file name')
parser.add_argument('-i','--it', type=int, help='Number of iterations')
args = parser.parse_args()

exp_name = args.exp
assert isinstance(exp_name, str), "Experiment name invalid or missing"
run = args.run
assert isinstance(run, int), "Run number invalid or missing"
fileName = args.file
assert isinstance(args.file, str), "Output file name missing"
it = args.it
assert isinstance(it, int), "Not a correct number of iterations"

event_start = 0
event_break = 5

# The ROI size can be defined here for now
N2 = 40
M2 = 40
#ROI = [ Start:Start+N2-1,Start:Start+M2-1 ]
Xstart = 20
Ystart = 20
roi = [Ystart,Ystart+N2,Xstart,Xstart+M2]

# Photon conversion
ADU = 325

# Extracting the data 
ds = ps.DataSource('exp='+exp_name+':run='+str(run)+':idx')
run = ds.runs().next()
det = ps.Detector('andor')
times = run.times()
env = ds.env()
eventTotal = len(times)

evt = run.event(times[0])    
eventTotal = len(times)

# Have to enter iterations manually for now
it  = 5000

# calculate array size per core and partition arrays
N = it/int(numWorkers) + 1
print('Runing: '+ str(it) + ' iterations with a number of tasks per core of: '+ str(N) )

assert( len(times) >= numWorkers )

allJobs = np.arange(len(times))
jobChunks = np.array_split(allJobs, numWorkers)
myChunk = jobChunks[rank]
myJobs = allJobs[myChunk[0]:myChunk[-1]+1]

f = h5.File('parallel_speckle.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
dset = f.create_dataset('speckle_pattern', (numWorkers,), dtype='i')

for i in range(len(times)):
    if i % numWorkers == rank:
        evt = run.event(times[i])
        dset[i]  = det.raw(evt) - det.pedestals(evt)       

f.close() 


# Discriminate shots based on k_average and crop data
Data = []
k_av = 0.08

#for zz in range(speckle_patterns.shape[0]):
for zz in range(35000):
    Essai = speckle_patterns[zz]
    Essai[Essai<20] = 0
    k_ave = Essai.sum()/(N2*M2)/ADU
    if k_ave < k_av:
        continue
    else:
        Data.append(speckle_patterns[zz,roi[0]:roi[1],roi[2]:roi[3]])

del speckle_patterns
Data = np.asarray(Data)

# Clipping the data
Data[Data<20] = 0
Data[Data>2000] = 0

print('There are '+ str(Data.shape[0]) +' shots with a k average below '+ str(k_av) )

assert Data.shape[0] > it, 'Number of discriminated shots is inferior to iterations'

# initialize photon map
photonMap = np.zeros((N,N2,M2))
rawImages = np.zeros((N,N2,M2))
dropGuess = np.zeros((N,N2,M2))

# initialize event numbers (-1 means bad event)
eventNums = np.ones(N)
#eventNums =[]

# Grid to create Gaussian photons
xIm = np.linspace(0,M2-1,M2)
yIm = np.linspace(0,N2-1,N2)
xIm,yIm = np.meshgrid(xIm,yIm)

###### let's start the main droplet loop over shots ######

tic = time.clock()

for jj in range(it):
     
    # parallel processing over "good" events
    if (jj+1)%size!=rank: continue
    
    mask = Data[jj]>0
    label_im, nb_labels = ndimage.label(mask) 
    print('Running iteration number '+ str(jj)+ ' on core ' +str(rank))
    print('There are '+ str(nb_labels)+' droplets in event '+str(jj) + ' and '+ str(np.round(np.sum(Data[jj])/325))+\
          ' total photons.')
    
    for i in range(nb_labels):

        drop = dr.Droplet(Data[jj],label_im, i+1, xIm, yIm, 0.5)
        drop.optimize_photons()
        drop.map_photons()
        if i<10:
            print(str(i)+' of '+str(nb_labels)+ ' has ' + str(drop.numPhotons)+' photons')
            print('Found '+str(np.sum(drop.photonMap))+' of '+str(drop.numPhotons)+' photons')
        if drop.numPhotons>np.sum(drop.photonMap):
            break
           
        # update full photon map
        photonMap[n2,:,:] += drop.photonMap
        dropGuess[n2,:,:] += drop.dropGuess
        # label this event as good
        
    eventNums[jj] = jj
    toc = time.clock()
    print('time elapsed for this shot: ' +str(toc - tic))
    
        
# How to tell the log that this process is done
print('Process '+ str(rank)+' finished.')

# How to wait for all the core to be finished ??
# Gathering and saving the data on the 0th core

photonMap = comm.gather(photonMap)
dropGuess = comm.gather(dropGuess)
eventNums = comm.gather(eventNums)

if rank==0:

    # concatenate arrays from all processes
    photonMap = np.concatenate(photonMap, axis=0)
    dropGuess = np.concatenate(dropGuess, axis=0)
    eventNums = np.concatenate(eventNums, axis=None)
      
    f = h5.File( fileName +'.h5','w')
    f.create_dataset('photonMap',data=photonMap)
    f.create_dataset('dropGuess',data=dropGuess)
    f.create_dataset('eventNums',data=eventNums)
    f.close()


