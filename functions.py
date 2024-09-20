import numpy
import scipy

import MDAnalysis as mda
import matplotlib.pylab as plt


def perform_search_time(XYZ_UNI, COFF, INIT, END, STRIDE, BOX_const=False):
    beads = XYZ_UNI.select_atoms('all')
    cont_list = list()
    #loop over traj
    for i,ts in enumerate(XYZ_UNI.trajectory[INIT:END:STRIDE]):
        if BOX_const == True:
            if i==0:
                nsearch = mda.lib.NeighborSearch.AtomNeighborSearch(beads, box=XYZ_UNI.dimensions)
        else:
            nsearch = mda.lib.NeighborSearch.AtomNeighborSearch(beads, box=XYZ_UNI.dimensions)
        cont_list.append([nsearch.search(i, COFF, level='A')  for i in beads])
    return cont_list

def xyz_to_uni(xyz, box=None, traj_name_output=None):
    
    # reading input xyz and box dimensions
    xyz_u = mda.Universe(xyz)
    n_monomers = len(xyz_u.atoms)
    n_frames = len(xyz_u.trajectory)

    if not isinstance(box, str):
        xyz_box_sigle = mda.lib.mdamath.triclinic_box(box[0],box[1],box[2])[:3]
        xyz_box = numpy.array([xyz_box_sigle for i in range(n_frames)])
        angles = list(mda.lib.mdamath.triclinic_box(box[0],box[1],box[2])[3:])
    else:
        xyz_box = numpy.loadtxt(box)
        angles = [90.,90.,90.]
        
    # new Universe init
    new_u = mda.Universe.empty(n_atoms=n_monomers,
                               n_residues=n_monomers,
                               n_segments=1,
                               atom_resindex=numpy.arange(n_monomers),
                               residue_segindex=[1]*n_monomers,
                               trajectory=True)
    
    # init new empty traj
    new_traj = numpy.empty((n_frames, n_monomers, 3))
    
    # loop over empty frame to insert the new frames
    for i,ts in enumerate(xyz_u.trajectory):
        empty_frame = numpy.empty((n_monomers, 3))
        for j,pos in enumerate(xyz_u.atoms.positions):
            empty_frame[j] = pos
        new_traj[i] = empty_frame
        
    # add the dimension of the box to the new traj
    new_u.load_new(new_traj, format=mda.coordinates.memory.MemoryReader)
    for s,snap in enumerate(new_u.trajectory):
        box_dim_tmp = numpy.pad(xyz_box[s], (0, 3), 'constant') + numpy.array([0.,0.,0.]+angles)
        new_u.trajectory[snap.frame].dimensions = box_dim_tmp
        
    return new_u


def local_dynamics(list_sum):
    particle = [i for i in range(numpy.shape(list_sum)[1])]
    ncont_tot = list()
    nn_tot = list()
    num_tot = list()
    den_tot  = list()
    for p in particle:
        ncont = list()
        nn = list()
        num = list()
        den = list()
        for frame in range(len(list_sum)):
            if frame == 0:
                ncont.append(0)
                nn.append(0)
            else:
                # se il set di primi vicini cambia totalmente, l'intersezione è lunga 1 ovvero la bead self
                # vale anche se il numero di primi vicini prima e dopo cambia
                if len(list(set(list_sum[frame-1][p]) & set(list_sum[frame][p])))==1:
                    # se non ho NN lens è 0
                    if len(list(set(list_sum[frame-1][p])))==1 and len(set(list_sum[frame][p]))==1:
                        ncont.append(0)
                        nn.append(0)
                        num.append(0)
                        den.append(0)
                    # se ho NN lo metto 1
                    else:
                        ncont.append(1)
                        nn.append(len(list_sum[frame][p])-1)
                        num.append(1)
                        den.append(len(list_sum[frame-1][p])-1+len(list_sum[frame][p])-1)    
                else:
                    # contrario dell'intersezione fra vicini al frame f-1 e al frame f
                    c_diff = set(list_sum[frame-1][p]).symmetric_difference(set(list_sum[frame][p]))
                    ncont.append(len(c_diff)/(len(list_sum[frame-1][p])-1+len(list_sum[frame][p])-1))
                    nn.append(len(list_sum[frame][p])-1)
                    num.append(len(c_diff))
                    den.append(len(list_sum[frame-1][p])-1+len(list_sum[frame][p])-1)
        num_tot.append(num)
        den_tot.append(den)
        ncont_tot.append(ncont)
        nn_tot.append(nn)
    return ncont_tot, nn_tot, num_tot, den_tot

def check(value, b_chunk):
    if b_chunk[0] <= value < b_chunk[1]:
        return True
    return False


def savgol_filter_mod(ncont_tot,polyorder,window,plot=True, ylim=None, xticks=None, xticks_l=None,yticks=None, yticks_l=None, xunit='$\mu$', windows_study=[10,50,100,150],polyorder_study=[2,4,6]):
    ncont_rolling = list()
    particle = [i for i in range(numpy.shape(ncont_tot)[0])]
    for p in particle:
        savgol_2_10 = scipy.signal.savgol_filter(ncont_tot[p], window_length=windows_study[0],polyorder=polyorder_study[0])
        savgol_2_50 = scipy.signal.savgol_filter(ncont_tot[p], window_length=windows_study[1],polyorder=polyorder_study[0])
        savgol_2_100 = scipy.signal.savgol_filter(ncont_tot[p], window_length=windows_study[2],polyorder=polyorder_study[0])
        savgol_2_150 = scipy.signal.savgol_filter(ncont_tot[p], window_length=windows_study[3],polyorder=polyorder_study[0])

        savgol_4_10 = scipy.signal.savgol_filter(ncont_tot[p], window_length=windows_study[0],polyorder=polyorder_study[1])
        savgol_4_50 = scipy.signal.savgol_filter(ncont_tot[p], window_length=windows_study[1],polyorder=polyorder_study[1])
        savgol_4_100 = scipy.signal.savgol_filter(ncont_tot[p], window_length=windows_study[2],polyorder=polyorder_study[1])
        savgol_4_150 = scipy.signal.savgol_filter(ncont_tot[p], window_length=windows_study[3],polyorder=polyorder_study[1])


        savgol_6_10 = scipy.signal.savgol_filter(ncont_tot[p], window_length=windows_study[0],polyorder=polyorder_study[2])
        savgol_6_50 = scipy.signal.savgol_filter(ncont_tot[p], window_length=windows_study[1],polyorder=polyorder_study[2])
        savgol_6_100 = scipy.signal.savgol_filter(ncont_tot[p], window_length=windows_study[2],polyorder=polyorder_study[2])    
        savgol_6_150 = scipy.signal.savgol_filter(ncont_tot[p], window_length=windows_study[3],polyorder=polyorder_study[2])

        savgol = scipy.signal.savgol_filter(ncont_tot[p], window_length=window,polyorder=polyorder)
        ncont_rolling.append(savgol[int(window/2):-int(window/2)])
        
        if p%100==0 and plot:
            fig, ax = plt.subplots(1,4, figsize=(16,4), dpi=500)
            fig.suptitle(r'Bead ID '+str(p), size=20)
            ax[0].set_title("window: " +str(windows_study[0]), fontsize=20)
            ax[0].plot(savgol_2_10, c='green', label='poly order '+str(polyorder_study[0]))
            ax[0].plot(savgol_4_10, c='blue', label='poly order '+str(polyorder_study[1]))
            ax[0].plot(savgol_6_10, c='red', label='poly order '+str(polyorder_study[2]))
            ax[0].plot(ncont_tot[p], c='gray',lw=0.5, alpha=0.8)
            ax[0].set_ylim(ylim)
            ax[0].set_ylabel(r'$\delta_b^{\tau}$', size=20)
            ax[0].set_xlabel(r't ['+xunit+'s]', size=20)
            ax[0].set_xticks(xticks, fontsize=20)
            ax[0].set_xticklabels(xticks_l, fontsize=20)
            ax[0].set_yticks(yticks, fontsize=20)
            ax[0].set_yticklabels(yticks_l, fontsize=20)
            ax[0].legend()

            ax[1].set_title("window: " +str(windows_study[1]), fontsize=20)
            ax[1].plot(savgol_2_50, c='green', label='poly order '+str(polyorder_study[0]))
            ax[1].plot(savgol_4_50, c='blue', label='poly order '+str(polyorder_study[1]))
            ax[1].plot(savgol_6_50, c='red', label='poly order '+str(polyorder_study[2]))
            ax[1].plot(ncont_tot[p], c='gray',lw=0.1, alpha=0.8)
            ax[1].set_ylim(ylim)
            ax[1].set_xlabel(r't ['+xunit+'s]', size=20)
            ax[1].set_xticks(xticks, fontsize=20)
            ax[1].set_xticklabels(xticks_l, fontsize=20)
            ax[1].set_yticks([], fontsize=20)
            ax[1].set_yticklabels([], fontsize=20)
            ax[1].legend()

            ax[2].set_title("window: " +str(windows_study[2]), fontsize=20)        
            ax[2].plot(savgol_2_100, c='green', label='poly order '+str(polyorder_study[0]))
            ax[2].plot(savgol_4_100, c='blue',label='poly order '+str(polyorder_study[1]))
            ax[2].plot(savgol_6_100, c='red', label='poly order '+str(polyorder_study[2]))
            ax[2].plot(ncont_tot[p], c='gray',lw=0.1, alpha=0.8)
            ax[2].set_ylim(ylim)
            ax[2].set_xlabel(r't ['+xunit+'s]', size=20)
            ax[2].set_xticks(xticks, fontsize=20)
            ax[2].set_xticklabels(xticks_l, fontsize=20)
            ax[2].set_yticks([], fontsize=20)
            ax[2].set_yticklabels([], fontsize=20)
            ax[2].legend()

            ax[3].set_title("window: " +str(windows_study[3]), fontsize=20)
            ax[3].plot(savgol_2_150, c='green', label='poly order '+str(polyorder_study[0]))
            ax[3].plot(savgol_4_150, c='blue', label='poly order '+str(polyorder_study[1]))
            ax[3].plot(savgol_6_150, c='red', label='poly order '+str(polyorder_study[2]))
            ax[3].plot(ncont_tot[p], c='gray',lw=0.1, alpha=0.8)
            ax[3].set_ylim(ylim)
            ax[3].set_xlabel(r't ['+xunit+'s]', size=20)
            ax[3].set_xticks(xticks, fontsize=20)
            ax[3].set_xticklabels(xticks_l, fontsize=20)
            ax[3].set_yticks([], fontsize=20)
            ax[3].set_yticklabels([], fontsize=20)        
            ax[3].legend()

    #         for coff in range(len(dynamic_coff)):
    #             ax[0].hlines(dynamic_coff[coff], INIT/STRIDE, END/STRIDE, colors='black', linewidth=2, linestyle=':')
    #             ax[1].hlines(dynamic_coff[coff], INIT/STRIDE, END/STRIDE, colors='black', linewidth=2, linestyle=':')
    #             ax[2].hlines(dynamic_coff[coff], INIT/STRIDE, END/STRIDE, colors='black', linewidth=2, linestyle=':')
    #             ax[3].hlines(dynamic_coff[coff], INIT/STRIDE, END/STRIDE, colors='black', linewidth=2, linestyle=':')
    #             ax[3].text(END/STRIDE+END/STRIDE*0.066,dynamic_coff[coff],'c'+str(coff),fontsize=18)        
            plt.tight_layout()
    return ncont_rolling

# Add properties based on properties list
def add_properties(prop_dict):
    prop = {"name" : [p[0] for p in prop_dict["Properties"]],\
            "type" : [p[1] for p in prop_dict["Properties"]],\
            "col" : [p[2] for p in prop_dict["Properties"]]}

    properties_l = list()
    for p in range(len(prop["name"])):
        properties_l.append(str(prop["name"][p])+\
                        ":"+str(prop["type"][p])+\
                        ":"+str(prop["col"][p]))
    return properties_l

# Add field in xyz comment line
def xyz_extender(prop_dict, i):
    ext_prop = 'Lattice="'+str(' '.join([str(item) for sublist in prop_dict["Lattice"][i] for item in sublist]))+\
            '" Origin="'+str(' '.join([str(item) for item in prop_dict["Origin"]]))+\
                '" Properties='+str(':'.join(add_properties(prop_dict)))
    return ext_prop

def arrowed_spines(
        ax,
        x_width_fraction=0.05,
        x_height_fraction=0.05,
        lw=None,
        ohg=0.3,
        locations=('bottom right', 'left up'),
        **arrow_kwargs
):

    # set/override some default plotting parameters if required
    arrow_kwargs.setdefault('overhang', ohg)
    arrow_kwargs.setdefault('clip_on', False)
    arrow_kwargs.update({'length_includes_head': True})

    # axis line width
    if lw is None:
        # FIXME: does this still work if the left spine has been deleted?
        lw = ax.spines['left'].get_linewidth()

    annots = {}

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # get width and height of axes object to compute
    # matching arrowhead length and width
    fig = ax.get_figure()
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = x_width_fraction * (ymax-ymin)
    hl = x_height_fraction * (xmax-xmin)

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    for loc_str in locations:
        side, direction = loc_str.split(' ')
        assert side in {'top', 'bottom', 'left', 'right'}, "Unsupported side"
        assert direction in {'up', 'down', 'left', 'right'}, "Unsupported direction"

        if side in {'bottom', 'top'}:
            if direction in {'up', 'down'}:
                raise ValueError("Only left/right arrows supported on the bottom and top")

            dy = 0
            head_width = hw
            head_length = hl

            y = ymin if side == 'bottom' else ymax

            if direction == 'right':
                x = xmin
                dx = xmax - xmin
            else:
                x = xmax
                dx = xmin - xmax

        else:
            if direction in {'left', 'right'}:
                raise ValueError("Only up/downarrows supported on the left and right")
            dx = 0
            head_width = yhw
            head_length = yhl

            x = xmin if side == 'left' else xmax

            if direction == 'up':
                y = ymin
                dy = ymax - ymin
            else:
                y = ymax
                dy = ymin - ymax


        annots[loc_str] = ax.arrow(x, y, dx, dy, fc='k', ec='k', lw = lw,
                 head_width=head_width, head_length=head_length, **arrow_kwargs)

    return annots
    
