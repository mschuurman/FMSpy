"""
Routines for reading input files and writing log files.
"""
import os
import sys
import h5py
import ast as ast
import shutil as shutil
import numpy as np
import nomad.core.glbl as glbl
import nomad.core.wavefunction as wavefunction
import nomad.core.trajectory as trajectory
import nomad.core.surface as surface
import nomad.integrals.integral as integral
import nomad.integrals.centroid as centroid

np.set_printoptions(threshold = np.inf)
tkeys       = ['traj', 'poten', 'grad', 'coup', 'hessian',
               'dipole', 'tr_dipole', 'sec_mom', 'atom_pop']
bkeys       = ['pop', 'energy', 'auto', 'spawn']
dump_header = dict()
dump_format = dict()
tfile_names = dict()
bfile_names = dict()

#
def archive_simulation(wfn, integrals, file_name=None):
    """Documentation to come"""

    # default is to use file name from previous write
    if file_name is not None:
        glbl.paths['chkpt_file'] = file_name.strip()

    # if this is the first time we're writing to the archive,
    # create the bundle data set and record the time-independent
    # bundle definitions
    if not os.path.isfile(glbl.paths['chkpt_file']):
        create(glbl.paths['chkpt_file'], wfn, integrals)

    # pull the time from the wave function to uniquely timestamp
    # entries
    time = wfn.time

    # open checkpoint file
    chkpt = h5py.File(glbl.paths['chkpt_file'], 'a', libver='latest')

    # write the wave function to file
    write_wavefunction(chkpt, wfn, time)

    # write the integral information to file
    write_integral(chkpt, integrals, time)

    # close the chkpt file
    chkpt.close()
    return

#
def retrieve_simulation(time=None, file_name=None, key_words=False):
    """Dochumentation to come"""

    # default is to use file name from previous write
    if file_name is not None:
        glbl.paths['chkpt_file'] = file_name.strip()

    # open chkpoint file
    chkpt = h5py.File(glbl.paths['chkpt_file'], 'r', libver='latest')

    if key_words:
        read_keywords(chkpt)

    # read wave function information, including trajectories
    wfn = read_wavefunction(chkpt, time)

    # update the wfn specific data
    ints = read_integral(chkpt, time)
    if ints is not None:
        ints.update(wfn)

    # close the checkpoint file
    chkpt.close()

    return [wfn, ints]

#
def merge_simulations(file_names=None, new_file=None):
    """Documentation to come"""

    if file_names is None:
        sys.exit('No files to merge. Exiting...')

    if new_file is None:
        sys.exit('No target file for merge named. Exiting...')

    for i in range(len(file_names)):
        if not os.path.isfile(file_names[i]):
            sys.exit('Cannot merge '+str(file_names[i])+
                     ': File does not exist. Exiting...')

    # we will copy the first file in file_names to the target,
    # then merge all subsequent files into that on
    shutil.copy(file_names[0], new_file)    
    target = h5py.File(new_file, 'a', libver='latest')   

    wcnt = sum('wavefunction' in grp for grp in target.keys())
    icnt = sum('integral' in grp for grp in target.keys())
    for i in range(1,len(file_names)):
        chkpt = h5py.File(file_names[i], 'r', libver='latest')    
        for grp in chkpt:
            if 'wavefunction' in grp:
                chkpt.copy(grp, target, name='wavefunction.'+str(wcnt))
                wcnt += 1
            elif 'integral' in grp:
                chkpt.copy(grp, target, name='integral.'+str(icnt))
                icnt += 1
        chkpt.close()    

    target.close()

# 
def time_steps(chkpt=None, grp_name=None, file_name=None):
    """Documentation to come"""
    # if file handle is None, get file stream by opening
    # file, file_name
    if chkpt is None and file_name is not None:
        chkpt = h5py.File(file_name.strip(), 'r', libver='latest') 

    if grp_name is None:
        grp_name = 'wavefunction.0'

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

    # if opening from file name, close file when done
    if chkpt is None and file_name is not None:
        chkpt.close()

    return steps

#
def update_basis(time, parent, child, file_name=None, name=0):
    """ Update information regarding new basis functions """
    
    basis_name = 'basis.'+str(name)
    dset       = basis_name+'/adapt_log'

    if file_name is not None:
        chkpt = h5py.File(file_name.strip(), 'r', libver='latest')
    else:
        chkpt = h5py.File(glbl.paths['chkpt_file'], 'a', libver='latest')

    data_row = package_basis(time, parent, child) 
    chkpt[basis_name].attrs['current_row'] += 1
    chkpt[dset][chkpt[basis_name].attrs['current_row']] = data_row

    return

#
def retrieve_basis(file_name=None, name=0):

    basis_name = 'basis.'+str(name)
    dset       = basis_name+'/adapt_log'

    # default is to use file name from previous write
    if file_name is not None:
        glbl.paths['chkpt_file'] = file_name.strip()

    # open chkpoint file
    chkpt = h5py.File(glbl.paths['chkpt_file'], 'r', libver='latest')

    basis_data = chkpt[dset][0:chkpt[basis_name].attrs['current_row']]

    chkpt.close()

    return basis_data

#------------------------------------------------------------------------------------
#
# Should not be called outside the module
#
#------------------------------------------------------------------------------------
def create(file_name, wfn, ints):
    """Creates a new checkpoint file."""
    # create chkpoint file
    chkpt = h5py.File(file_name, 'w', libver='latest')

    # save the contents of glbl.py
    write_keywords(chkpt)

    # create basis group
    create_basis(chkpt, wfn, name=0)

    # wfn group -- row information
    create_wfn(chkpt, wfn, name=0)

    # integral group -- row information
    create_int(chkpt, ints, name=0)

    # close following initialization
    chkpt.close()

# 
def create_basis(chkpt, wfn, name=0):
    """ Creates a new basis group, with suffix 'name' """
   
    basis_name = 'basis.'+str(name)

    if basis_name in chkpt.keys():
        raise ValueError('wavefunction='+wfn_name+' already exists.'+
                         'Continuing...') 
    else:
        chkpt.create_group(basis_name)
        chkpt[basis_name].attrs['n_traj']      = wfn.n_traj()
        chkpt[basis_name].attrs['current_row'] = -1
        chkpt[basis_name].attrs['n_rows']      = 100

        # create the 'spawn.log' table 
        dset     = basis_name+'/adapt_log'
        dshape   = (chkpt[basis_name].attrs['n_rows'], 13)
        chkpt.create_dataset(dset, dshape, dtype=float, compression="gzip")

        # add initial trajectories with a 'born time' of 0
        for i in range(wfn.n_traj()):
            data_row  = package_basis(wfn.time, wfn.traj[i], wfn.traj[i]) 
            chkpt[basis_name].attrs['current_row'] += 1         
            chkpt[dset][chkpt[basis_name].attrs['current_row']] = data_row
        
    return

#
def create_wfn(chkpt, wfn, name=0):
    """Creates a new wavefunction group, with suffix 'name' """

    wfn_name = 'wavefunction.'+str(name)
    
    if wfn_name in chkpt.keys():
        raise ValueError('wavefunction='+wfn_name+' already exists.'+
                         'Continuing...') 
    else:
        chkpt.create_group(wfn_name)
        chkpt[wfn_name].attrs['current_row'] = -1
        chkpt[wfn_name].attrs['n_rows']      = 0

    return

#
def create_int(chkpt, ints, name=0):
    """Creates a new integral group, with suffix 'name' """

    int_name = 'integral.'+str(name)

    if int_name in chkpt.keys():
        raise ValueError('integral='+int_name+' already exists.'+
                         'Continuing...')
    else:
        chkpt.create_group(int_name)
        chkpt[int_name].attrs['current_row']     = -1
        chkpt[int_name].attrs['n_rows']          = 0

        # integral group -- time independent obj properties
        chkpt[int_name].attrs['kecoef']            = ints.kecoef
        chkpt[int_name].attrs['ansatz']            = ints.ansatz
        chkpt[int_name].attrs['numerical']         = ints.numerical
        chkpt[int_name].attrs['hermitian']         = ints.hermitian
        chkpt[int_name].attrs['require_centroids'] = ints.require_centroids

    return

#
def write_keywords(chkpt):
    """Writes the contents of glbl to the checkpoint file. This 
       is only done once upon the creation of the file"""
    
    #loop over the dictionaries in glbl
    for keyword_section in glbl.sections.keys():
    
        #if module/class objects, skip
        if keyword_section == 'modules':
            continue
    
        grp_name = 'keywords_'+keyword_section
        chkpt.create_group(grp_name)
        for keyword in glbl.sections[keyword_section].keys():
            write_keyword(chkpt, grp_name, keyword, 
                          glbl.sections[keyword_section][keyword])

    return

#
def write_keyword(chkpt, grp, kword, val):
    """Write a keyword to simulation archive"""
    
    # try writing variable to h5py attribute using native format 
    try:
        chkpt[grp].attrs[kword] = val
        return

    except:
        pass

    # that fails, write as a string
    try:
        #..and if an array, preserve commas
        if isinstance(val, np.ndarray):
            sval = ','.join([val[i] for i in range(len(val))])
        else:
            sval = str(val)
        d_type = h5py.special_dtype(vlen=str) 
        chkpt[grp].attrs.create(kword, sval, dtype=d_type)

    except Exception as e: 
        print("Failed to write keyword:"+str(kword)+" = val:"+str(val)+
              " -- "+str(e)+"\n")

    return


def read_keywords(chkpt):
    """Read keywords from archive file"""

    # open chkpoint file
    chkpt = h5py.File(glbl.paths['chkpt_file'], 'r', libver='latest')

    #loop over the dictionaries in glbl
    for keyword_section in glbl.sections.keys():

        #if module/class objects, skip
        if keyword_section == 'modules':
            continue

        grp_name = 'keywords_'+keyword_section
        for keyword in glbl.sections[keyword_section].keys():
            val = read_keyword(chkpt, grp_name, keyword)
            try:
                glbl.sections[keyword_section][keyword] = val
            except Exception as e:     
                print("Failed to set keyword:"+str(keyword)+" -- "+str(e)+"\n")

    chkpt.close()

    return

def read_keyword(chkpt, grp, kword):
    """Read a particular keyword attribute"""

    # try writing variable to h5py attribute using native format 
    try:
        val = chkpt[grp].attrs[kword]
        return convert_value(kword, val)

    except Exception as e:     
        print("Failed to read keyword:"+str(kword)+" -- "+str(e)+"\n")

    return


def convert_value(kword, val):
    """Converts a string value to NoneType, bool, int, float or string."""

    cval = val
   
    # if we can't interpret this as a list, return the string
    if str(cval).find(',',0) == -1:
        return cval

    # we have some items that are lists of strings which are converted
    # to simple strings. Try to interpret as list, and if that fails, return
    # the value unchanged
    try:
        cval = ast.literal_eval(val)
        if isinstance(cval, list):
            cval = np.ndarray(cval)
            return cval
    except ValueError:
        pass 
    try:
        cval = str(cval).split(',')
    except ValueError:
        pass

    # else just a string and return as-is
    return cval

def write_wavefunction(chkpt, wfn, time, name=0):
    """Documentation to come"""
    wfn_data = package_wfn(wfn)
    n_traj   = wfn.n_traj()
    n_blk    = default_blk_size(time)
    resize   = False
    wfn_name = 'wavefunction.'+str(name)

    # if wfn doesn't exist, add it on the fly
    if wfn_name not in chkpt.keys():
        create_wfn(chkpt, name=name)

    # update the current row index (same for all data sets)
    chkpt[wfn_name].attrs['current_row'] += 1
    current_row = chkpt[wfn_name].attrs['current_row']

    if current_row == chkpt[wfn_name].attrs['n_rows']:
        resize = True
        chkpt[wfn_name].attrs['n_rows'] += n_blk
    n_rows = chkpt[wfn_name].attrs['n_rows']

    # first write items with time-independent dimensions
    for data_label in wfn_data.keys():
        dset = wfn_name+'/'+data_label

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


def write_integral(chkpt, integral, time, name=0):
    """Documentation to come"""
    int_data = package_integral(integral, time)
    n_blk    = default_blk_size(time)
    resize   = False
    int_name = 'integral.'+str(name)

    # if integral doesn't exist, add it on the fly
    if int_name not in chkpt.keys():
        create_int(chkpt, integral, name=name)

    # update the current row index (same for all data sets)
    chkpt[int_name].attrs['current_row'] += 1
    current_row = chkpt[int_name].attrs['current_row']

    if current_row == chkpt[int_name].attrs['n_rows']:
        resize   = True
        chkpt[int_name].attrs['n_rows'] += n_blk
    n_rows = chkpt[int_name].attrs['n_rows']

    # first write items with time-independent dimensions
    for data_label in int_data.keys():
        dset = int_name+'/'+data_label

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
        for i in range(len(integral.centroids)):
            for j in range(i):
                 if integral.centroids[i][j] is not None:
                     write_centroid(chkpt, integral.centroids[i][j], time)


def write_trajectory(chkpt, traj, time, name=0):
    """Documentation to come"""
    # open the trajectory file
    t_data   = package_trajectory(traj, time)
    t_label  = str(traj.label)
    n_blk    = default_blk_size(time)
    resize   = False
    grp_name = 'wavefunction.'+str(name)

    # if trajectory group already exists, just append current
    # time information to existing datasets
    t_grp = grp_name+'/'+t_label

    if t_grp in chkpt:

        chkpt[t_grp].attrs['current_row'] += 1
        current_row = chkpt[t_grp].attrs['current_row']

        if current_row == chkpt[t_grp].attrs['n_rows']:
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
        current_row                       = 0
        chkpt[t_grp].attrs['current_row'] = current_row
        chkpt[t_grp].attrs['n_rows']      = n_blk
        n_rows                            = chkpt[t_grp].attrs['n_rows']

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


def write_centroid(chkpt, cent, time, name=0):
    """Documentation to come"""
    # open the trajectory file
    c_data  = package_centroid(cent, time)
    c_label = str(cent.label)
    n_blk   = default_blk_size(time)
    resize  = False
    grp_name = 'integral.'+str(name)

    # if trajectory group already exists, just append current
    # time information to existing datasets
    c_grp = grp_name+'/'+c_label

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
        current_row                       = 0
        chkpt[c_grp].attrs['current_row'] = current_row
        chkpt[c_grp].attrs['n_rows']      = n_blk
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


def read_wavefunction(chkpt, time, name=0):
    """Documentation to come"""

    nstates  = glbl.properties['n_states']
    widths   = glbl.properties['crd_widths']
    masses   = glbl.properties['crd_masses']
    dim      = len(widths)
    wfn_name = 'wavefunction.'+str(name)
    int_name = 'integral.'+str(name)
    kecoef   = chkpt[int_name].attrs['kecoef'] #indicative of wrongess.
                                                 #trajectory should be purged of kecoef...

    # check that we have the desired time:
    read_row = get_time_index(chkpt, wfn_name, time)

    if read_row is None:
        ValueError('time='+str(time)+' requested, but not in checkpoint file')
        return None

    # create the wavefunction object to hold the data
    wfn = wavefunction.Wavefunction()

    # dimensions of these objects are not time-dependent
    wfn.time    = chkpt[wfn_name+'/time'][read_row,0]

    for label in chkpt[wfn_name]:

        #print("time = "+str(time)+" label="+str(label))
        if (label=='time' or label=='pop' or label=='energy'):
            continue

        t_grp = wfn_name+'/'+label
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

    return wfn

def read_integral(chkpt, time, name=0):
    """Documentation to come"""

    nstates  = glbl.properties['n_states']
    widths   = glbl.properties['crd_widths']
    dim      = len(widths)
    int_name = 'integral.'+str(name)   

    ansatz   = glbl.methods['ansatz']
    numerics = glbl.methods['integral_eval'] 
    kecoef   = chkpt[int_name].attrs['kecoef'] 

    # check that we have the desired time:
    read_row = get_time_index(chkpt, int_name, time)

    if read_row is None:
        raise ValueError('time='+str(time)+' requested, but not in checkpoint file')
        return None

    ints = integral.Integral(kecoef, ansatz, numerics)

    if ints.require_centroids:
        for label in chkpt[int_name]:

            if label == 'time':
                continue

            c_grp = int_name+'/'+label
            c_row = get_time_index(chkpt, c_grp, time)

            if c_row is None:
                continue

            new_cent = centroid.Centroid(nstates=nstates, dim=dim, width=widths)
            read_centroid(chkpt, new_cent, c_grp, c_row)
            ints.add_centroid(new_cent)

    return ints


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

    new_traj.state  = int(state)
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
    momt    = chkpt[c_grp+'/momentum'][c_row]

    new_cent.parents = [int(i) for i in parent]
    new_cent.states  = [int(i) for i in states]

    idi              = max(new_cent.parents)
    idj              = min(new_cent.parents)
    new_cent.label   = -((idi * (idi - 1) // 2) + idj + 1)

    new_cent.update_pes_info(pes)
    new_cent.pos = new_cent.pes.get_data('geom')
    new_cent.mom = momt


def get_time_index(chkpt, grp_name, time):
    """Documentation to come"""
    time_vals = time_steps(chkpt=chkpt, grp_name=grp_name)

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

#
def package_basis(time, parent, child):
    """Record when and from whom new basis functions are spawned"""

    basis_data = np.zeros(13, dtype=float)
    basis_data[[0,1,2]]   = [time, 
                            parent.last_spawn[child.state],
                            parent.exit_time[child.state]]
    basis_data[[3,4,5,6]] = [parent.label,      parent.state,
                            child.label,        child.state]
    basis_data[[7,8]]     = [parent.kinetic(),   child.kinetic()] 
    basis_data[[9,10]]    = [parent.potential(), child.potential()]
    basis_data[[11,12]]   = [parent.classical(), child.classical()]

    return basis_data

#
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
        glbl     = np.concatenate((np.array([traj.parent, traj.state, traj.gamma,
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
    # let's just keep this to small default size: 25
    # need to look into optimizing this more
    return 25


#-----------------------------------------------------------------------------
#
#  printing routines 
#
#-----------------------------------------------------------------------------
def generate_data_formats():
    """Initialized all the output format descriptors."""
    global dump_header, dump_format, tfile_names, bfile_names

    nst   = glbl.properties['n_states']
    ncrd  = len(glbl.properties['crd_widths'])
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

    # spawn table
    lenst        = 7
    arr1  = ('time(entry)','time(spawn)', 'time(exit)')
    arr2  = ('parent','state','child','state')
    arr3  = ('ke(parent)','ke(child)',
             'pot(parent)','pot(child)',
             'total(parent)','total(child)')
    bfile_names['spawn'] = 'spawn.dat'
    dump_header['spawn'] = (''.join([arr1[i].rjust(acc1) for i in range(len(arr1))]) +
                            ''.join([arr2[i].rjust(7) for i in range(len(arr2))]) + 
                            ''.join([arr3[i].rjust(acc2) for i in range(len(arr3))])+'\n')
    dump_format['spawn'] = ('{:12.4f}{:12.4f}{:12.4f}{:7.1f}{:7.1f}{:7.1f}{:7.1f}' +
                            '{:16.8f}{:16.8f}{:16.8f}{:16.8f}' +
                            '{:16.8f}{:16.8f}\n')

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

    return

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

