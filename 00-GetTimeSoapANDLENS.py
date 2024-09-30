from functions import *
from functions_new import *

import numpy as np
import os

import matplotlib.pylab as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle

import seaborn as sns
from seaborn import heatmap
from seaborn import kdeplot

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.signal import savgol_filter

import SOAPify
from SOAPify import SOAPclassification, calculateTransitionMatrix, normalizeMatrixByRow, getSOAPSettings
import SOAPify.HDF5er as HDF5er

from MDAnalysis import Universe as mdaUniverse
from MDAnalysis import transformations
from MDAnalysis.tests.datafiles import TPR, XTC
from SOAPify.HDF5er import MDA2HDF5
from os import path

from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster


###### 1. INPUT Parameters

XYZ_DIR = 'INPUT_files/Wave/'  # insert DIR Path containing XYZ Input Files
XYZ_OUTDIR = 'Wave/'  # insert OUTPUT DIR Path (to store LENS and TimeSOAP Output files)
NAME = 'wave_mod' # NAME.hdf5
GROUP = "/Trajectories/"+NAME
MASK = False #be sure that mask used to create .hdf5 is the same used here

# Default: False (if you want to calculate LENS and TimeSOAP)
LOAD_HDF5 = False
LOAD_HDF5SOAP = False
LOAD_SOAP = False
LOAD_TimeSOAP = False
LOAD_LENS = False

TRJ_filename = NAME + ".xtc"  # or .lammpsdump
TOP_filename = topo = NAME + ".gro"  # or .data

#XYZ_ORIG = [0, 0, 0]
COFF=40  # CUTOFF radius: be sure that the same coff is used to create soaps and lenses   
wantedTrajectory = slice(0, None, None)  # TRJ we want to analyze
wantedUniverseAtoms = "type C"  # atom types to keep in our local environment (to save in .hdf5)

eq=0

SOAPnmax = 8
SOAPlmax = 8
SOAPatomMask = "type C"  # atom types to use to get LENS and TimeSOAP 

###### 2. Create HDF5 File from TOP and TRJ Files

if LOAD_HDF5:

    if MASK:
        HDF5_TRJ = NAME + "mask.hdf5"
        HDF5_TRJ_nomask = NAME + ".hdf5"
        with h5py.File(XYZ_DIR + HDF5_TRJ, "r") as trajFile:
            tgroup = trajFile[GROUP]
            universe = HDF5er.createUniverseFromSlice(tgroup, wantedTrajectory)
    else:
        HDF5_TRJ = NAME + ".hdf5"
        HDF5_TRJ_nomask = NAME + ".hdf5"
        with h5py.File(XYZ_DIR + HDF5_TRJ, "r") as trajFile:
            tgroup = trajFile[GROUP]
            universe = HDF5er.createUniverseFromSlice(tgroup, wantedTrajectory)

else:
    mycwd = os.getcwd()
    os.chdir(XYZ_DIR)
    createHDF5(TRJ_filename, TOP_filename, wantedUniverseAtoms)  # Create HDF5 File (if not present)
    os.chdir(mycwd)
    HDF5_TRJ = NAME + ".hdf5"
    with h5py.File(XYZ_DIR + HDF5_TRJ, "r") as trajFile:
        tgroup = trajFile[GROUP]
        universe = HDF5er.createUniverseFromSlice(tgroup, wantedTrajectory)

    if MASK:  # TODO: change HDF5 filename in mask.hdf5
        HDF5_TRJ = NAME + "mask.hdf5"
        HDF5_TRJ_nomask = NAME + ".hdf5"
        with h5py.File(XYZ_DIR + HDF5_TRJ, "r") as trajFile:
            tgroup = trajFile[GROUP]
            universe = HDF5er.createUniverseFromSlice(tgroup, wantedTrajectory)

    else:
        HDF5_TRJ = NAME + ".hdf5"
        HDF5_TRJ_nomask = NAME + ".hdf5"
        with h5py.File(XYZ_DIR + HDF5_TRJ, "r") as trajFile:
            tgroup = trajFile[GROUP]
            universe = HDF5er.createUniverseFromSlice(tgroup, wantedTrajectory)

nAtoms = len(universe.atoms)

###### 3. Create HDF5SOAP File

if LOAD_SOAP:
    pass

else:

    mycwd = os.getcwd()
    os.chdir(XYZ_DIR)
    prepareSOAP(HDF5_TRJ_nomask, NAME, COFF, SOAPnmax, SOAPlmax, SOAPatomMask)
    os.chdir(mycwd)

###### 4. Get TimeSOAP

if LOAD_TimeSOAP:
    tsoap = np.load(XYZ_DIR + 'tsoap.npy')
    print(np.shape(tsoap))
else:

    if MASK:
        with h5py.File(XYZ_DIR + NAME + '.hdf5', 'r') as f:
            mask = f[GROUP + "Trajectory"][0, :, 2] > 12

        with h5py.File(XYZ_DIR + NAME + 'soap.hdf5', 'r') as f:
            ds = f["SOAP/" + NAME]
            fillSettings = getSOAPSettings(ds)
            X = ds[:][:, mask]

    else:
        with h5py.File(XYZ_DIR + NAME + 'soap.hdf5', 'r') as f:
            ds = f["SOAP/" + NAME]
            fillSettings = getSOAPSettings(ds)
            X = ds[:, :, :]

    X = SOAPify.fillSOAPVectorFromdscribe(
        X[:], **fillSettings)

    X = SOAPify.normalizeArray(X)
    np.savez(XYZ_DIR + 'X_normalized.npz', name1=X)

#    v_soap = np.load(XYZ_DIR + 'X_normalized.npz')['name1'][windowToUSE]
#    print(np.shape(v_soap))

    nAtoms, tsoap, dtSOAP = getTimeSOAP(XYZ_DIR + NAME + 'soap.hdf5', NAME)

np.savez(XYZ_DIR+"time_dSOAP.npz", name1=tsoap)

###### 4. Get LENS

if LOAD_LENS:
    LENS = np.load(XYZ_DIR+'lens.npy')
else:
    neigCounts = SOAPify.analysis.listNeighboursAlongTrajectory(universe, cutOff=COFF)
    LENS, nn, *_ = SOAPify.analysis.neighbourChangeInTime(neigCounts)
    np.save(XYZ_DIR+"lens", LENS)

