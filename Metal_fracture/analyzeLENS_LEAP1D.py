
#%%
import SOAPify.HDF5er as HDF5er
import SOAPify
import h5py
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from multiprocessing.pool import ThreadPool as Pool
from seaborn import kdeplot

cutoff = 8

#%%
wantedTrajectory = slice(0, None, 1)
trajFileName = "planeSlip.hdf5"
trajAddress = "/Trajectories/planeSlip"
with h5py.File(trajFileName, "r") as trajFile:
    tgroup = trajFile[trajAddress]
    universe = HDF5er.createUniverseFromSlice(tgroup, wantedTrajectory)



#READ LEAP
LENS_normalized = np.load("/home/mattia/Desktop/Fracture/tests_mattia/Cu/NPT_tensile/test_01/remapX_1dt/planeSlip/timeseries_analysis/LEAP-1D/leap_1D_radialdistance.npy")
cluster_thresholds = [0]  # Modifica le soglie come desideri
colors=['w']
LENS_normalized[:,0] = LENS_normalized[:,2]  
# Trova i cluster in base alle soglie
clusters = np.zeros_like(LENS_normalized, dtype=int)
for i, threshold in enumerate(cluster_thresholds):
    clusters[LENS_normalized >= threshold] = i + 1


# Crea un dizionario per archiviare le informazioni sui cluster
data_KM = {}
for i, threshold in enumerate(cluster_thresholds):
    data_KM[i + 1] = {}
    data_KM[i + 1]["elements"] = LENS_normalized[clusters == i + 1]
    data_KM[i + 1]["min"] = np.min(data_KM[i + 1]["elements"])
    data_KM[i + 1]["max"] = np.max(data_KM[i + 1]["elements"])

fig, axes = plt.subplots(
    1, figsize=(5, 3), dpi=500, 
)

 
for flns in LENS_normalized:

    axes.plot(
        # [x + 138850 for x in range(len(LENS_normalized[0]))],
        range(len(LENS_normalized[0])),
        flns,
        color="k",
        linewidth=0.01,
        # alpha=0.9,
    )

axes.set_xlim(0, len(LENS_normalized[0])-1)
axes.set_ylabel(r'|LEAP|',weight='bold',size=20)
axes.set_xlabel("t [ns]",weight='bold', size=20)

newtiks= ['138.85', '138.87', '138.89', '138.91', '138.93', '138.95',]
axes.set_xticklabels(newtiks,fontsize=12)
axes.set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontsize=12)
axes.set_ylim([0,1])

