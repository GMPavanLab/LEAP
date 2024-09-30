
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

# %%
#open npz
LENS = np.load('lens.npz')['arr_0']
nn = np.load('lens.npz')['arr_1']
nAtoms = np.shape(LENS)[0]
print(nAtoms, 'atoms')

# %%
#FILTERING############################################àà
def signaltonoise(a: np.array, axis, ddof):
    """Given an array, retunrs its signal to noise value of its compontens"""
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20 * np.log10(abs(np.where(sd == 0, 0, m / sd)))
# %%
atom = 30
savGolPrint = np.arange(LENS[atom].shape[0])
windows = [5, 10, 15]
polyorders = [2, 3, 4]
fig, axes = plt.subplots(len(windows), sharey=True)

for ax, window_length in zip(axes, windows):
    windowToPlot = slice(window_length // 2, -window_length // 2)
    ax.plot(LENS[atom], label=f"Atom {atom}")
    for polyorder in polyorders:
        savgol = savgol_filter(
            LENS, window_length=window_length, polyorder=polyorder, axis=-1
        )

        sr_ratio = signaltonoise(savgol[:, windowToPlot], axis=-1, ddof=0)

        ax.plot(
            savGolPrint[windowToPlot],
            savgol[atom, windowToPlot],
            label=f"Filtered w={window_length}, p={polyorder} s/n={np.mean(sr_ratio):.2f}",
        )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

# %%
windows = np.arange(10, 10, 5)
polyorders = [2, 2,2 ]￼
fig, ax = plt.subplots(1)


for polyorder in polyorders:
    meansrRatios = np.empty((len(windows), 2))
    for i, window_length in enumerate(windows):
        windowToPlot = slice(window_length // 2, -window_length // 2)
        savgol = savgol_filter(
            LENS, window_length=window_length, polyorder=polyorder, axis=-1
        )
        sr_ratio = signaltonoise(savgol[:, windowToPlot], axis=-1, ddof=0)
        meansrRatios[i] = [window_length, np.mean(sr_ratio)]
    ax.plot(meansrRatios[:, 0], meansrRatios[:, 1], label=f"p={polyorder}")
ax.set_xlabel("Window Size", fontsize=20)
ax.set_ylabel("S/N [dB]", fontsize=20)

_ = ax.legend()


# %%
#normalize the data
minLens = np.min(LENS)
print(minLens)
maxLens = np.max(LENS)
print(maxLens)
LENS_normalized = (LENS-minLens)/(maxLens-minLens)  # NORMALIZATION
np.save('LENS_normalized.npy',LENS_normalized)

# %%
LENS_normalized[:,0] = LENS_normalized[:,4]
# %%
cluster_thresholds = [0.33,0.59]  # Modifica le soglie come desideri
colors=['lightsalmon','red']
labelsize = 15
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
    1,figsize=(5, 3), dpi=500, 
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

# hist = axes[1].hist(
#     LENS_normalized.reshape((-1)),
#     alpha=0.7,
#     color="gray",
#     bins="doane",
#     density=True,
#     orientation="horizontal",
# )

# kde = kdeplot(
#     y=LENS_normalized.reshape((-1)),
#     bw_adjust=1.5,
#     linewidth=2,
#     color="black",
#     # gridsize=500,
#     gridsize=2 * (hist[1].shape[0] - 1),
#     ax=axes[1],
# )

# # axes[0].set_title('lens "trajectories"')
# # axes[0].set_xlim(0, len(LENS_normalized[0]))
# axes[1].set_title("Density of\nlens values")
# height = np.max(hist[0])

for col_idx, (cname, data) in enumerate(data_KM.items()):
    # center = delta+data['min']
    bin_size = data["max"] - data["min"]
    for i, ax in enumerate([axes]):
        ax.barh(
            y=data["min"],
            width=len(LENS_normalized[0]-1) if i == 0 else height,
            height=bin_size,
            align="edge",
            alpha=0.25,
            color=colors[col_idx],
            linewidth=2,
        )
        ax.axhline(y=data["min"], c=colors[col_idx], linewidth=2, linestyle="--")
        if i == 1:
            ax.text(
                -0.25,
                data["min"] + bin_size / 2.0,
                f"c{cname}",
                va="center",
                ha="right",
                fontsize=18,
            )

for cname, data in data_KM.items():
    num_atoms_in_cluster = len(data["elements"])
    print(f"clusters {cname}: {num_atoms_in_cluster}")
for cname, data in data_KM.items():
    print(f"Cluster {cname}:")
    for key, value in data.items():
        print(f"  {key}: {value}")
axes.set_xlim(0, len(LENS_normalized[0])-1)
axes.set_ylabel(r'LENS',weight='bold',size=20)
axes.set_xlabel("t [ns]",weight='bold', size=20)

newtiks= ['138.85', '138.87', '138.89', '138.91', '138.93', '138.95',]
axes.set_xticklabels(newtiks,fontsize=12)
axes.set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontsize=12)
axes.set_ylim([0,1])
# labels = [item.get_text() for item in axes[0].get_xticklabels()]
# labels[2] = 1220
# axes[0].set_xticklabels(labels)
# Set the new tick positions
# axes[0].set_xticklabels(new_ticks)

# %%

# def prepareData(x, /):
#     """prepares an array from shape (atom,frames) to  (frames,atom)"""
#     shape = x.shape
#     toret = np.empty((shape[1], shape[0]), dtype=x.dtype)
#     for i, atomvalues in enumerate(x):
#         toret[:, i] = atomvalues
#     return toret


# # classyfing by knoing the min/max of the clusters
# def classifying(x, classDict):
#     toret = np.ones_like(x, dtype=int) * (len(classDict) - 1)
#     # todo: sort  by max and then classify
#     minmax = [[cname, data["max"]] for cname, data in data_KM.items()]
#     minmax = sorted(minmax, key=lambda x: -x[1])
#     # print(minmax)
#     for cname, myMax in minmax:
#         toret[x < myMax] = int(cname)
#     return toret


# # classifying(filteredLENS,data_KM)
# classifiedNormalizedLENS = classifying(LENS_normalized, data_KM)



# %%

clusters_tSOAP = np.load('clusters_tSOAP.npy')

def export(wantedTrajectory):
    # as a function, so the ref universe should be garbage collected correctly
    with h5py.File(trajFileName, "r") as trajFile, open(
        "outClassifiedLENS_3clusters.xyz", "w"
    ) as xyzFile:
        from MDAnalysis.transformations import fit_rot_trans

        tgroup = trajFile[trajAddress]
        ref = HDF5er.createUniverseFromSlice(tgroup, [0])
        nAt= len(ref.atoms)
        ref.add_TopologyAttr("mass", [1] * nAt)
        # load antohter univer to avoid conatmination in the main one
        exportuniverse = HDF5er.createUniverseFromSlice(tgroup, wantedTrajectory)
        exportuniverse.add_TopologyAttr("mass", [1] * nAt)
        #exportuniverse.trajectory.add_transformations(fit_rot_trans(exportuniverse, ref))
        HDF5er.getXYZfromMDA(
            xyzFile,
            exportuniverse,
            # framesToExport=wantedTrajectory,
            #allFramesProperty='Origin="-40 -40 -40"',
            #LENS=LENS.T,
            normalizedLENS=LENS_normalized.T,
            proposedClassification=clusters.T,
            #proposedClassification_tSOAP=clusters_tSOAP.T,
        )
        universe.trajectory


export(wantedTrajectory)
# %%
# find all  iDs of particles that have a signal higher than threshold
threshold = 0.82
idFounds = np.where(np.any(LENS_normalized > threshold, axis=1))[0]
#print for ovito
print(len(idFounds))
output_string = ' || '.join([f'ParticleIndex == {id}' for id in idFounds])
print(output_string)



# %%
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# %%
en = np.loadtxt('log.dat')
energy = en[:, 2]
time = en[:,0]/1000
# Normalizza l'energia tra 0 e 1
min_energy = np.min(energy)
max_energy = np.max(energy)
normalized_energy = (energy - min_energy) / (max_energy - min_energy)
 
maxima_indices = np.where(normalized_energy == np.max(normalized_energy))[0]
maxima_indices_lens = np.where(LENS_normalized == np.max(LENS_normalized))[1]
# Plotta l'energia
 
fig, ax1 = plt.subplots(dpi=200, figsize=(4,2.5))
 
# Plotta FLNS sovrapposto all'energia
color='k'
for flns in LENS_normalized:
    ax1.plot(time,flns, color=color, linewidth=0.02,alpha=0.3)

ax1.set_ylabel('Lens signal',color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xlabel('Time [ps]')

color='r'
ax2 = ax1.twinx() 
ax2.plot(time,energy, color=color, linewidth=2)
ax2.set_ylabel('Potential energy',color=color)
ax2.tick_params(axis='y', labelcolor=color)


# for idx in maxima_indices:
#     plt.axvline(x=idx, color="r", linestyle='--', label=f'Massimo {idx}')
# for idx2 in maxima_indices_lens:
#     plt.axvline(x=idx2, color="b", linestyle='--', label=f'Massimo {idx2}')
 

# %%
#READ TSOAP ##########################################################################
#LOAD
tSOAP = np.load('tSOAP.npz')['tSOAP']

#aggiungo riga e traspongo
print(tSOAP.shape)
new_row = np.zeros((1, tSOAP.shape[1]))
tSOAP = np.vstack((new_row, tSOAP))
tSOAP = tSOAP.T
print(tSOAP.shape)

#normalize the data
min = np.min(tSOAP)
print(min)
max = np.max(tSOAP)
print(max)
tSOAP_normalized = (tSOAP-min)/(max-min)  # NORMALIZATION


#%%
nAtoms = tSOAP.shape[0]



# %%

cluster_thresholds = [0.0,0.4,0.6]  # Modifica le soglie come desideri

# Trova i cluster in base alle soglie
clusters = np.zeros_like(tSOAP_normalized, dtype=int)
for i, threshold in enumerate(cluster_thresholds):
    clusters[tSOAP_normalized >= threshold] = i + 1

#%%
# Crea un dizionario per archiviare le informazioni sui cluster
data_KM = {}
for i, threshold in enumerate(cluster_thresholds):
    data_KM[i + 1] = {}
    data_KM[i + 1]["elements"] = tSOAP_normalized[clusters == i + 1]
    data_KM[i + 1]["min"] = np.min(data_KM[i + 1]["elements"])
    data_KM[i + 1]["max"] = np.max(data_KM[i + 1]["elements"])



# %%
fig, axes = plt.subplots(
    1, 2, figsize=(6, 3), dpi=200, width_ratios=[3, 1.2], sharey=True
)

for i in range(nAtoms):
    axes[0].plot(
        range(len(tSOAP_normalized[0])),
        tSOAP_normalized[i],
        color="k",
        linewidth=0.01,
        # alpha=0.9,
    )
hist = axes[1].hist(
    tSOAP_normalized.reshape((-1)),
    alpha=0.7,
    color="gray",
    bins="doane",
    density=True,
    orientation="horizontal",
)
kde = kdeplot(
    y=tSOAP_normalized.reshape((-1)),
    bw_adjust=1.5,
    linewidth=2,
    color="black",
    # gridsize=500,
    gridsize=2 * (hist[1].shape[0] - 1),
    ax=axes[1],
)
axes[0].set_title('tSOAP "trajectories"')
axes[0].set_xlim(0, len(tSOAP_normalized[0]))
axes[1].set_title("Density of\ntSOAP values")
height = np.max(hist[0])
for cname, data in data_KM.items():
    # center = delta+data['min']
    bin_size = data["max"] - data["min"]
    for i, ax in enumerate(axes):
        ax.barh(
            y=data["min"],
            width=len(tSOAP_normalized[0]) if i == 0 else height,
            height=bin_size,
            align="edge",
            alpha=0.25,
            # color=colors[idx_m],
            linewidth=2,
        )
        ax.axhline(y=data["min"], c="red", linewidth=2, linestyle=":")
        if i == 1:
            ax.text(
                -0.25,
                data["min"] + bin_size / 2.0,
                f"c{cname}",
                va="center",
                ha="right",
                fontsize=18,
            )



# %%
#find time of max en pot, lens, tSOAP######################################################################
#in picoseconds
peakEn = en[np.argmax(en[:,2]),0]/1000

peakLENS = np.unravel_index(np.argmax(LENS_normalized, axis=None), LENS_normalized.shape)[1]/10

peaktSOAP = 19.2


# %%
#LEAP
from functions import *
from functions_new import *


#%%
djoint_d = np.dstack((np.transpose(tSOAP_normalized),np.transpose(LENS_normalized)))
np.shape(djoint_d)
djoint_d_fl = np.array(djoint_d).reshape(np.shape(djoint_d)[0]*np.shape(djoint_d)[1],np.shape(djoint_d)[2])
djoint_d_fl.shape
# %%
_ = plt.scatter(djoint_d_fl[:,0],djoint_d_fl[:,1], marker='.', color='black', alpha=0.1)
# plt.plot([0,1],[0,1],color='red',linestyle='--')
plt.xlabel('tSOAP', fontsize=20)
plt.ylabel('LENS', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.xlim((0,1))
# plt.ylim((0,1))
# %%
_, ax = plt.subplots(figsize=(4, 4), dpi=200, tight_layout=True)
H, x, y = np.histogram2d(djoint_d_fl[:,0], djoint_d_fl[:,1], bins=300)
xcenters = (x[:-1] + x[1:]) / 2

ycenters = (y[:-1] + y[1:]) / 2

ax.pcolormesh(xcenters, ycenters, np.log(H).T, cmap=plt.get_cmap('jet'), alpha=1)
ax.set_xlabel(r'$\tau$SOAP', weight='bold',size=16)
ax.set_ylabel('LENS', weight='bold',size=16)

for side in ['right','top','left','bottom']:
    ax.spines[side].set_visible(False)
    
for side in ['bottom','right','top','left']:
    ax.spines[side].set_linewidth(1)

ax.tick_params(
axis='both',          # changes apply to the x-axis
which='major',      # both major and minor ticks are affected
bottom=False,
left=False,      
labelleft=False,
labelbottom=False) 
annots = arrowed_spines(ax, locations=('bottom right', 'left up')) 

# %%


