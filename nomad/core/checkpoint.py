"""
Routines for reading input files and writing log files.
"""
import os
import h5py
import numpy as np
import nomad.core.glbl as glbl
import nomad.integrals.integral as integral
import nomad.core.wavefunction as wavefunction
import nomad.core.trajectory as trajectory
import nomad.core.surface as surface


np.set_printoptions(threshold = np.inf)
tkeys       = ['traj', 'poten', 'grad', 'coup', 'hessian',
               'dipole', 'tr_dipole', 'secm', 'apop']
bkeys       = ['pop', 'energy', 'auto']
dump_header = dict()
dump_format = dict()
tfile_names = dict()
bfile_names = dict()


def archive_simulation(wfn, integrals=None, time=None, file_name=None):
    """Documentation to come"""
    write(wfn, file_name=file_name, time=time)
    if integrals is not None:
        write(integrals, file_name=file_name, time=time)


def retrieve_simulation(wfn, integrals=None, time=None, file_name=None):
    """Dochumentation to come"""
    read(wfn, file_name=file_name, time=time)
    if integrals is not None:
        read(integrals, file_name=file_name, time=time)


def write(data_obj, file_name=None, time=None):
    """Documentation to come"""
    # default is to use file name from previous write
    if file_name is not None:
        glbl.chkpt_file = file_name.strip()

    # if this is the first time we're writing to the archive,
    # create the bundle data set and record the time-independent
    # bundle definitions
    if not os.path.isfile(glbl.chkpt_file):
        if isinstance(data_obj, wavefunction.Wavefunction):
            create(glbl.chkpt_file, data_obj)
        else:
            raise TypeError('chkpt file must be created with wavefunction object.')

    # open checkpoint file
    chkpt = h5py.File(glbl.chkpt_file, 'a', libver='latest')

    # this definition of time over-rules all others as far as writing
    # data is concerned.
    if time is None:
        time = data_obj.time

    #------------------------------------------
    # this data get written for all simulations
    if isinstance(data_obj, wavefunction.Wavefunction):
        write_wavefunction(chkpt, data_obj, time)

    elif isinstance(data_obj, integral.Integral):
        write_integral(chkpt, data_obj, time)
    else:
        raise TypeError('data_obj: '+str(data_obj)+' is not recognized by checkpoint.write')

    chkpt.close()


def read(data_obj, file_name, time=None):
    """Reads the checkpoint file.

    Called when an old checkpoint file is used to populate the
    contents of a bundle, if no time given, read last bundle.
    """
    # string name of checkpoint file
    glbl.chkpt_file = file_name.strip()

    # open chkpoint file
    chkpt = h5py.File(glbl.chkpt_file, 'r', libver='latest')

    if isinstance(data_obj, wavefunction.Wavefunction):

        read_wavefunction(chkpt, data_obj, time)

    # if this is an integral objects, it's going to want to load the centroid data
    # from file
    elif isinstance(data_obj, integral.Integral):
        read_integral(chkpt, data_obj, time)
    else:
        raise TypeError('data_obj: '+str(data_obj)+' is not recognized by checkpoint.read')

    # close checkpoint file
    chkpt.close()


def time_steps(grp_name, file_name=None):
    """Documentation to come"""
    # default is to use file name from previous write
    if file_name is not None:
        glbl.chkpt_file = file_name.strip()

    # open chkpoint file
    chkpt = h5py.File(glbl.chkpt_file, 'r', libver='latest')

    # if the group name is in the checkpoint file, return
    # the associated time array
    if grp_name in chkpt:
        if 'current_row' in chkpt[grp_name].attrs:
            current_row = chkpt[grp_name].attrs['current_row'] + 1
        else:
            current_row = len(chkpt[grp_name+'/time'][:])
        steps = chkpt[grp_name+'/time'][:current_row, 0]
    #else abort
    else:
        raise ValueError('grp_name: '+str(grp_name)+' not present in checkpoint file')

    chkpt.close()

    return steps


def generate_data_formats(ncrd, nst):
    """Initialized all the output format descriptors."""
    global dump_header, dump_format, tfile_names, bfile_names

    ncart = 3         # assumes expectation values of transition/permanent dipoles in
                      # cartesian coordinates
    natm  = max(1,int(ncrd / ncart)) # dirty -- in case we have small number of n.modes
    dstr  = ('x', 'y', 'z')
    acc1  = 12
    acc2  = 16

    # ******************* dump formats *******************************

    # ----------------- trajectory data --------------------------------
    # trajectory output
    arr1 = ['{:>12s}'.format('    x' + str(i+1)) for i in range(ncrd)]
    arr2 = ['{:>12s}'.format('    p' + str(i+1)) for i in range(ncrd)]
    tfile_names[tkeys[0]] = 'trajectory'
    dump_header[tkeys[0]] = ('Time'.rjust(acc1) + ''.join(arr1) +
                             ''.join(arr2) + 'Phase'.rjust(acc1) +
                             'Re[Amp]'.rjust(acc1) + 'Im[Amp]'.rjust(acc1) +
                             'Norm[Amp]'.rjust(acc1) + 'State'.rjust(acc1) +
                             '\n')
    dump_format[tkeys[0]] = ('{:12.4f}'+
                             ''.join('{:12.6f}' for i in range(2*ncrd+5))+
                             '\n')

    # potential energy
    arr1 = ['{:>16s}'.format('potential.' + str(i)) for i in range(nst)]
    tfile_names[tkeys[1]] = 'poten'
    dump_header[tkeys[1]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[1]] = ('{:12.4f}' +
                             ''.join('{:16.10f}' for i in range(nst)) + '\n')

    # gradients
    arr1 = ['            x' + str(i+1) for i in range(ncrd)]
    tfile_names[tkeys[2]] = 'grad'
    dump_header[tkeys[2]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[2]] = ('{0:>12.4f}' +
                             ''.join('{' + str(i) + ':14.8f}'
                                     for i in range(1, ncrd+1)) + '\n')

    # coupling
    arr1 = ['{:>12s}'.format('coupling.' + str(i)) for i in range(nst)]
    arr2 = ['{:>12s}'.format('c * v .' + str(i)) for i in range(nst)]
    tfile_names[tkeys[3]] = 'coup'
    dump_header[tkeys[3]] = ('Time'.rjust(acc1) + ''.join(arr1) +
                             ''.join(arr2) + '\n')
    dump_format[tkeys[3]] = ('{:12.4f}' +
                             ''.join('{:12.5f}' for i in range(2*nst)) + '\n')


    # ---------------------- interface data --------------------------------
    # permanent dipoles
    arr1 = ['{:>12s}'.format('dip_st' + str(i) + '.' + dstr[j])
            for i in range(nst) for j in range(ncart)]
    tfile_names[tkeys[5]] = 'dipole'
    dump_header[tkeys[5]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[5]] = ('{:12.4f}' +
                             ''.join('{:12.5f}'
                                     for i in range(nst*ncart)) + '\n')
    # transition dipoles
    arr1 = ['  td_s' + str(j) + '.s' + str(i) + '.' + dstr[k]
            for i in range(nst) for j in range(i) for k in range(ncart)]
    ncol = int(nst*(nst-1)*ncart/2+1)
    tfile_names[tkeys[6]] = 'tr_dipole'
    dump_header[tkeys[6]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[6]] = ('{:12.4f}' +
                             ''.join('{:12.5f}'
                                     for i in range(1, ncol)) + '\n')

    # second moments
    arr1 = ['   sec_s' + str(i) + '.' + dstr[j] + dstr[j]
            for i in range(nst) for j in range(ncart)]
    tfile_names[tkeys[7]] = 'sec_mom'
    dump_header[tkeys[7]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[7]] = ('{:12.4f}' +
                             ''.join('{:12.5f}'
                                     for i in range(nst*ncart)) + '\n')

    # atomic populations
    arr1 = ['    st' + str(i) + '_a' + str(j+1)
            for i in range(nst) for j in range(natm)]
    tfile_names[tkeys[8]] = 'atom_pop'
    dump_header[tkeys[8]] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format[tkeys[8]] = ('{:12.4f}' +
                             ''.join('{:10.5f}'
                                     for i in range(nst*natm)) + '\n')

    # ----------------- dump formats (wavefunction files) -----------------

    # adiabatic state populations
    arr1 = ['     state.' + str(i) for i in range(nst)]
    bfile_names['pop'] = 'n.dat'
    dump_header['pop'] = ('Time'.rjust(acc1) + ''.join(arr1) +
                             'Norm'.rjust(acc1) + '\n')
    dump_format['pop']  = ('{:12.4f}' +
                             ''.join('{:12.6f}' for i in range(nst)) +
                             '{:12.6f}\n')

    # the bundle energy
    arr1 = ('   potential(QM)', '     kinetic(QM)', '       total(QM)',
            '  potential(Cl.)', '    kinetic(Cl.)', '      total(Cl.)')
    bfile_names['energy'] = 'e.dat'
    dump_header['energy'] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format['energy'] = ('{:12.4f}' +
                             ''.join('{:16.10f}' for i in range(6)) + '\n')

    # autocorrelation function
    arr1 = ('      Re a(t)','         Im a(t)','         abs a(t)')
    bfile_names['auto'] = 'auto.dat'
    dump_header['auto'] = 'Time'.rjust(acc1) + ''.join(arr1) + '\n'
    dump_format['auto'] = ('{:12.4f}' +
                             ''.join('{:16.10f}' for i in range(3)) + '\n')

    # trajectory matrices
    tfile_names['hessian']   = 'hessian.dat'

    # bundle matrices
    bfile_names['t']         = 't.dat'
    bfile_names['v']         = 'v.dat'
    bfile_names['s']         = 's.dat'
    bfile_names['sdot']      = 'sdot.dat'
    bfile_names['h']         = 'h.dat'
    bfile_names['heff']      = 'heff.dat'
    bfile_names['t_overlap'] = 't_overlap.dat'


def print_traj_row(label, key, data):
    """Appends a row of data, formatted by entry 'fkey' in formats to
    file 'filename'."""
    filename = tfile_names[key] + '.' + str(label)

    if not os.path.isfile(filename):
        with open(filename, 'x') as outfile:
            outfile.write(dump_header[key])
            outfile.write(dump_format[key].format(*data))
    else:
        with open(filename, 'a') as outfile:
            outfile.write(dump_format[key].format(*data))


def print_traj_mat(time, key, mat):
    """Prints a matrix to file with a time label."""
    filename = tfile_names[key]

    with open(filename, 'a') as outfile:
        outfile.write('{:9.2f}\n'.format(time))
        outfile.write(np.array2string(mat,
                      formatter={'complex_kind':lambda x: '{: 15.8e}'.format(x)})+'\n')


def print_wfn_row(key, data):
    """Appends a row of data, formatted by entry 'fkey' in formats to
    file 'filename'."""
    filename = bfile_names[key]

    if not os.path.isfile(filename):
        with open(filename, 'x') as outfile:
            outfile.write(dump_header[key])
            outfile.write(dump_format[key].format(*data))
    else:
        with open(filename, 'a') as outfile:
            outfile.write(dump_format[key].format(*data))


def print_wfn_mat(time, key, mat):
    """Prints a matrix to file with a time label."""
    filename = bfile_names[key]

    with open(filename, 'a') as outfile:
        outfile.write('{:9.2f}\n'.format(time))
        outfile.write(np.array2string(mat,
                      formatter={'complex_kind':lambda x: '{: 15.8e}'.format(x)})+'\n')


#------------------------------------------------------------------------------------
#
# Should not be called outside the module
#
#------------------------------------------------------------------------------------
def create(file_name, wfn):
    """Creates a new checkpoint file."""
    # create chkpoint file
    chkpt = h5py.File(file_name, 'w', libver='latest')

    chkpt.create_group('wavefunction')
    chkpt['wavefunction'].attrs['current_row'] = -1
    chkpt['wavefunction'].attrs['n_rows']      = 0 

    chkpt.create_group('integral')
    chkpt['integral'].attrs['current_row']     = -1
    chkpt['integral'].attrs['n_rows']          = 0

    traj0 = wfn.traj[0]
    chkpt.create_group('simulation')
    chkpt['simulation'].attrs['nstates']  = traj0.nstates
    chkpt['simulation'].attrs['dim']      = traj0.dim
    chkpt['simulation'].attrs['widths']   = traj0.widths()
    chkpt['simulation'].attrs['masses']   = traj0.masses()
    chkpt['simulation'].attrs['kecoef']   = traj0.kecoef

    # close following initialization
    chkpt.close()


def write_wavefunction(chkpt, wfn, time):
    """Documentation to come"""
    wfn_data = package_wfn(wfn)
    n_traj   = wfn.n_traj()
    n_blk    = default_blk_size(time)
    resize   = False

    # update the current row index (same for all data sets)
    chkpt['wavefunction'].attrs['current_row'] += 1
    current_row = chkpt['wavefunction'].attrs['current_row']

    if current_row == chkpt['wavefunction'].attrs['n_rows']:
        resize = True
        chkpt['wavefunction'].attrs['n_rows'] += n_blk
    n_rows = chkpt['wavefunction'].attrs['n_rows']

    # first write items with time-independent dimensions
    for data_label in wfn_data.keys():
        dset = 'wavefunction/'+data_label

        if dset in chkpt:
            if resize:
                d_shape  = (n_rows,) + wfn_data[data_label].shape
                chkpt[dset].resize(d_shape)
            chkpt[dset][current_row] = wfn_data[data_label]

        # if this is the first time we're trying to write this bundle,
        # create a new datasets with reasonble default sizes
        else:
            d_shape   = (n_rows,) +  wfn_data[data_label].shape
            max_shape = (None,)   + wfn_data[data_label].shape
            d_type    = wfn_data[data_label].dtype
            chkpt.create_dataset(dset, d_shape, maxshape=max_shape, dtype=d_type, compression="gzip")
            chkpt[dset][current_row] = wfn_data[data_label]

    # now step through and write trajectories
    for i in range(n_traj):
        write_trajectory(chkpt, wfn.traj[i], time)


def write_integral(chkpt, integral, time):
    """Documentation to come"""
    int_data = package_integral(integral, time)
    n_blk    = default_blk_size(time)
    resize   = False

    # update the current row index (same for all data sets)
    chkpt['integral'].attrs['current_row'] += 1
    current_row = chkpt['integral'].attrs['current_row']

    if current_row == chkpt['integral'].attrs['n_rows']:
        resize   = True
        chkpt['integral'].attrs['n_rows'] += n_blk
    n_rows = chkpt['integral'].attrs['n_rows']

    # first write items with time-independent dimensions
    for data_label in int_data.keys():
        dset = 'integral/'+data_label

        if dset in chkpt:
            if resize:
                d_shape  = (n_rows,) + int_data[data_label].shape
                chkpt[dset].resize(d_shape)
            chkpt[dset][current_row] = int_data[data_label]

        # if this is the first time we're trying to write this bundle,
        # create a new datasets with reasonble default sizes
        else:
            d_shape   = (n_rows,) + int_data[data_label].shape
            max_shape = (None,)   + int_data[data_label].shape
            d_type    = int_data[data_label].dtype
            chkpt.create_dataset(dset, d_shape, maxshape=max_shape, dtype=d_type, compression="gzip")
            chkpt[dset][current_row] = int_data[data_label]

    # now step through centroids, if they're present
    if integral.require_centroids:
        for i in range(len(integral.centroid)):
            for j in range(i):
                 if integral.centroid[i][j] is not None:
                     write_centroid(chkpt, integral.centroid[i][j], time)


def write_trajectory(chkpt, traj, time):
    """Documentation to come"""
    # open the trajectory file
    t_data  = package_trajectory(traj, time)
    t_label = str(traj.label)
    n_blk   = default_blk_size(time)
    resize  = False

    # if trajectory group already exists, just append current
    # time information to existing datasets
    t_grp = 'wavefunction/'+t_label

    if t_grp in chkpt:

        chkpt[t_grp].attrs['current_row'] += 1
        current_row = chkpt[t_grp].attrs['current_row']

        if (current_row == chkpt[t_grp].attrs['n_rows']):
            resize = True
            chkpt[t_grp].attrs['n_rows'] += n_blk
        n_rows = chkpt[t_grp].attrs['n_rows']

        for data_label in t_data.keys():
            dset = t_grp+'/'+data_label
            if resize:
                d_shape  = (n_rows,) + t_data[data_label].shape
                chkpt[dset].resize(d_shape)

            chkpt[dset][current_row] = t_data[data_label]

    # if this is the first time we're trying to write this trajectory,
    # create a new data group, and new data sets with reasonble default sizes
    else:

        chkpt.create_group(t_grp)
        chkpt[t_grp].attrs['current_row'] = 0 
        chkpt[t_grp].attrs['n_rows']      = n_blk
        n_rows                            = chkpt[t_grp].attrs['n_rows']
        current_row                       = chkpt[t_grp].attrs['current_row'] 

        # store surface information from trajectory
        for data_label in t_data.keys():
            dset = t_grp+'/'+data_label
            d_shape   = (n_rows,) + t_data[data_label].shape
            max_shape = (None,)   + t_data[data_label].shape
            d_type    = t_data[data_label].dtype
            if d_type.type is np.unicode_:
                d_type = h5py.special_dtype(vlen=str)
            chkpt.create_dataset(dset, d_shape, maxshape=max_shape, dtype=d_type, compression="gzip")
            chkpt[dset][current_row] = t_data[data_label]


def write_centroid(chkpt, cent, time):
    """Documentation to come"""
    # open the trajectory file
    c_data  = package_centroid(cent, time)
    c_label = str(cent.label)
    n_blk   = default_blk_size(time)
    resize  = False

    # if trajectory group already exists, just append current
    # time information to existing datasets
    c_grp = 'integral/'+c_label

    if c_grp in chkpt:

        chkpt[c_grp].attrs['current_row'] += 1
        current_row = chkpt[c_grp].attrs['current_row']

        if current_row == chkpt[c_grp].attrs['n_rows']:
            resize = True
            chkpt[c_grp].attrs['n_rows'] += n_blk
        n_rows = chkpt[c_grp].attrs['n_rows']

        for data_label in c_data.keys():
            dset = c_grp+'/'+data_label
            if resize:
                d_shape  = (n_rows,) + c_data[data_label].shape
                chkpt[dset].resize(d_shape)

            chkpt[dset][current_row] = c_data[data_label]

    # if this is the first time we're trying to write this trajectory,
    # create a new data group, and new data sets with reasonble default sizes
    else:

        chkpt.create_group(c_grp)
        chkpt[c_grp].attrs['current_row'] = 0 
        chkpt[c_grp].attrs['n_rows']      = n_blk
        current_row                       = chkpt[c_grp].attrs['current_row']
        n_rows                            = chkpt[c_grp].attrs['n_rows']

        # store surface information from trajectory
        for data_label in c_data.keys():
            dset = c_grp+'/'+data_label
            d_shape   = (n_rows,) + c_data[data_label].shape
            max_shape = (None,)   + c_data[data_label].shape
            d_type    = c_data[data_label].dtype
            if d_type.type is np.unicode_:
                d_type = h5py.special_dtype(vlen=str)
            chkpt.create_dataset(dset, d_shape, maxshape=max_shape, dtype=d_type, compression="gzip")
            chkpt[dset][current_row] = c_data[data_label]


def read_wavefunction(chkpt, wfn, time):
    """Documentation to come"""
    nstates = chkpt['simulation'].attrs['nstates']
    dim     = chkpt['simulation'].attrs['dim']
    widths  = chkpt['simulation'].attrs['widths']
    masses  = chkpt['simulation'].attrs['masses']
    kecoef  = chkpt['simulation'].attrs['kecoef']

    # check that we have the desired time:

    read_row = get_time_index(chkpt, 'wavefunction', time)

    if read_row is None:
        raise ValueError('time='+str(time)+' requested, but not in checkpoint file')

    # dimensions of these objects are not time-dependent
    wfn.nstates = nstates
    wfn.time    = chkpt['wavefunction/time'][read_row,0]

    for label in chkpt['wavefunction']:

        if (label=='time' or label=='pop' or label=='energy'):
            continue

        t_grp = 'wavefunction/'+label
        t_row = get_time_index(chkpt, t_grp, time)

        if t_row is None:
            continue

        new_traj = trajectory.Trajectory(nstates, dim,
                                         width=widths,
                                         mass=masses,
                                         label=label,
                                         kecoef=kecoef)
        read_trajectory(chkpt, new_traj, t_grp, t_row)
        wfn.add_trajectory(new_traj.copy())


def read_integral(chkpt, integral, time):
    """Documentation to come"""
    nstates = chkpt['simulation'].attrs['nstates']
    dim     = chkpt['simulation'].attrs['dim']
    widths  = chkpt['simulation'].attrs['widths']
    masses  = chkpt['simulation'].attrs['masses']

    # check that we have the desired time:
    read_row = get_time_index(chkpt, 'integral', time)

    if read_row is None:
        raise ValueError('time='+str(time)+' requested, but not in checkpoint file')

    if integral.require_centroids:
        for label in chkpt['integral']:

            if label == 'time':
                continue

            c_grp = 'integral/'+label
            c_row = get_time_index(chkpt, c_grp, time)

            if c_row is None:
                continue

            new_cent = integral.Centroid(nstates=nstates, dim=dim, width=widths)
            read_centroid(chkpt, new_cent, c_grp, c_row)
            integral.add_centroid(new_cent)


def read_trajectory(chkpt, new_traj, t_grp, t_row):
    """Documentation to come"""
    # populate the surface object in the trajectory

    # set information about the trajectory itself
    data_row = chkpt[t_grp+'/glbl'][t_row]
    [parent, state, new_traj.gamma, amp_real, amp_imag] = data_row[0:5]

    pes = surface.Surface()
    for data_label in chkpt[t_grp].keys():
        if pes.valid_data(data_label):
            dset = chkpt[t_grp+'/'+data_label]
            pes.add_data(data_label, dset[t_row])

    # currently, momentum has to be read in separately
    momt    = chkpt[t_grp+'/momentum'][t_row]

    new_traj.set_state(int(state))
    new_traj.parent = int(parent)
    new_traj.update_amplitude(amp_real+1.j*amp_imag)
    new_traj.last_spawn = data_row[5:]

    new_traj.update_pes_info(pes)
    new_traj.update_x(new_traj.pes.get_data('geom'))
    new_traj.update_p(momt)


def read_centroid(chkpt, new_cent, c_grp, c_row):
    """Documentation to come"""

    # set information about the trajectory itself
    parent = [0.,0.]
    states = [0.,0.]
    [parent[0], parent[1], states[0], states[1]] = chkpt[c_grp+'/glbl'][c_row]

    # populate the surface object in the trajectory
    pes = surface.Surface()
    for data_label in chkpt[c_grp].keys():
        if pes.valid_data(data_label):
            dset = chkpt[c_grp+'/'+data_label]
            pes.add_data(data_label, dset[c_row])

    # currently, momentum has to be read in separately
    momt    = chkpt[c_grp+'/momentum'][t_row]

    new_cent.parents = int(parent)
    new_cent.states  = int(states)

    new_cent.update_pes_info(pes)
    new_cent.update_x(new_cent.pes.get_data('geom'))
    new_cent.update_p(momt)


def get_time_index(chkpt, grp_name, time):
    """Documentation to come"""
    time_vals = time_steps(grp_name)

    if time is None:
        return chkpt[grp_name].attrs['current_row']

    dt       = np.absolute(time_vals - time)
    read_row = np.argmin(dt)

    # this tolerance is arbitrary: check if the matched time
    # is further than 0.5 * timestep to the next closest times,
    # else we don't have a match
    match_chk = []
    if read_row > 0:
        match_chk.extend([time_vals[read_row]-time_vals[read_row-1]])
    if read_row < len(time_vals)-1:
        match_chk.extend([time_vals[read_row+1]-time_vals[read_row]])

    if dt[read_row] > 0.5*min(match_chk):
        read_row = None

    return read_row

def package_wfn(wfn):
    """Documentation to come"""
    # dimensions of these objects are not time-dependent
    wfn_data = dict(
        time   = np.array([wfn.time], dtype='float'),
        pop    = np.array(wfn.pop()),
        energy = np.array([wfn.pot_quantum(),   wfn.kin_quantum(),
                           wfn.pot_classical(), wfn.kin_classical()])
                    )

    return wfn_data


def package_integral(integral, time):
    """Documentation to come"""
    int_data = dict(
        time = np.array([time],dtype='float')
                    )
    return int_data


def package_trajectory(traj, time):
    """Documentation to come"""
    # time is not an element in a trajectory, but necessary to
    # uniquely tag everything
    traj_data = dict(
        time     = np.array([time],dtype='float'),
        glbl     = np.concatenate((np.array([traj.parent, traj.state(), traj.gamma,
                                 traj.amplitude.real, traj.amplitude.imag]),
                                 traj.last_spawn)),
        momentum = traj.p()
                    )

    # store everything about the surface
    for obj in traj.pes.avail_data():
        traj_data[obj] = traj.pes.get_data(obj)

    return traj_data


def package_centroid(cent, time):
    """Documentation to come"""
    cent_data = dict(
        time     = np.array([time],dtype='float'),
        glbl     = np.concatenate((cent.parents, cent.states)),
        momentum = cent.p()
                     )

    # last, store everything about the surface
    for obj in cent.pes.avail_data():
        cent_data[obj] = cent.pes.get_data(obj)

    return cent_data


def default_blk_size(time):
    """Documentation to come"""
    return 10
#    return int(2.1 * (glbl.propagate['simulation_time']-time) /
#                      glbl.propagate['default_time_step'])
