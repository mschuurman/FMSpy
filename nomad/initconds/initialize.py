"""
Routines for initializing dynamics calculations.
"""
import numpy as np
import scipy.linalg as sp_linalg
import nomad.core.glbl as glbl
import nomad.core.log as log
import nomad.core.trajectory as trajectory
import nomad.core.wavefunction as wavefunction
import nomad.core.surface as evaluate
import nomad.core.checkpoint as checkpoint


def init_wavefunction(master):
    """Initializes the trajectories."""
    # initialize the interface we'll be using the determine the
    # the PES. There are some details here that trajectories
    # will want to know about

    glbl.interface.init_interface()

    # now load the initial trajectories into the bundle
    if glbl.sampling['restart']:
        checkpoint.retrieve_simulation(master, integrals=glbl.master_int,
                                       time=glbl.sampling['restart_time'], file_name=glbl.chkpt_file)
        if glbl.sampling['restart_time'] != 0.:
            master0 = wavefunction.Wavefunction()
            checkpoint.retrieve_simulation(master0, integrals=None, time=0., file_name=glbl.chkpt_file)
            save_initial_wavefunction(master0)
        else:
            save_initial_wavefunction(master)
    else:
        # first generate the initial nuclear coordinates and momenta
        # and add the resulting trajectories to the bundle
        glbl.init_conds.set_initial_coords(master)

        # set the initial state of the trajectories in bundle. This may
        # require evaluation of electronic structure
        set_initial_state(master)

        # set the initial amplitudes of the basis functions
        set_initial_amplitudes(master)

        # add virtual basis functions, if desired
        if glbl.sampling['virtual_basis']:
            virtual_basis(master)

        # update the integrals
        glbl.master_int.update(master)

        # once phase space position and states of the basis functions
        # are set, update the potential information
        evaluate.update_pes(master)

        # update the couplings for all the trajectories
        for i in range(master.n_traj()):
            glbl.interface.evaluate_coupling(master.traj[i])

        # compute the hamiltonian matrix...
        glbl.master_mat.build(master, glbl.master_int)
        master.update_matrices(glbl.master_mat)

        # so that we may appropriately renormalize to unity
        master.renormalize()

        # this is the bundle at time t=0.  Save in order to compute auto
        # correlation function
        save_initial_wavefunction(master)

    # write the wavefunction to the archive
    if glbl.mpi['rank'] == 0:
        checkpoint.archive_simulation(master, integrals=glbl.master_int,
                                      time=master.time, file_name=glbl.chkpt_file)

    log.print_message('t_step', [master.time, glbl.propagate['default_time_step'],
                                      master.nalive])

    return master.time


#---------------------------------------------------------------------------
#
# Private routines
#
#----------------------------------------------------------------------------
def set_initial_state(master):
    """Sets the initial state of the trajectories in the bundle."""
    # initialize to the state with largest transition dipole moment
    if glbl.sampling['init_brightest']:

        # set all states to the ground state
        for i in range(master.n_traj()):
            master.traj[i].set_state(0)
            # compute transition dipoles
            evaluate.update_pes_traj(master.traj[i])

        # set the initial state to the one with largest t. dip.
        for i in range(master.n_traj()):
            if 'dipole' not in master.traj[i].pes.avail_data():
                raise KeyError('trajectory '+str(i)+
                               ': Cannot set state by transition moments - '+
                               'dipole not in pes.avail_data()')

            tr_dipole = master.traj[i].pes.get_data('dipole')
            tdip = np.array([np.linalg.norm(tr_dipole[:,0,j])
                             for j in range(1, glbl.propagate['n_states'])])
            log.print_message('general',
                             ['Initializing trajectory '+str(i)+
                              ' to state '+str(np.argmax(tdip)+1)+
                              ' | tr. dipople array='+np.array2string(tdip, \
                              formatter={'float_kind':lambda x: "%.4f" % x})])
            master.traj[i].set_state(np.argmax(tdip)+1)

    # use "init_state" to set the initial state
    elif len(glbl.sampling['init_states']) == master.n_traj():
        for i in range(master.n_traj()):
            master.traj[i].set_state(glbl.sampling['init_states'][i])

    else:
        raise ValueError('Ambiguous initial state assignment.')


def set_initial_amplitudes(master):
    """Sets the initial amplitudes."""
    # if init_amp_overlap is set, overwrite 'amplitudes' that was
    # set in nomad.input
    if glbl.nuclear_basis['init_amp_overlap']:
        origin = make_origin_traj()

        # Calculate the initial expansion coefficients via projection onto
        # the initial wavefunction that we are sampling
        ovec = np.zeros(master.n_traj(), dtype=complex)
        for i in range(master.n_traj()):
            ovec[i] = glbl.master_int.nuc_overlap(master.traj[i], origin)
        smat = np.zeros((master.n_traj(), master.n_traj()), dtype=complex)
        for i in range(master.n_traj()):
            for j in range(i+1):
                smat[i,j] = glbl.master_int.traj_overlap(master.traj[i],
                                                         master.traj[j])
                if i != j:
                    smat[j,i] = smat[i,j].conjugate()
        sinv = sp_linalg.pinvh(smat)
        glbl.nuclear_basis['amplitudes'] = np.dot(sinv, ovec)

    # if we didn't set any amplitudes, set them all equal -- normalization
    # will occur later
    elif len(glbl.nuclear_basis['amplitudes']) == 0:
        glbl.nuclear_basis['amplitudes'] = np.ones(master.n_traj(),dtype=complex)

    # if we don't have a sufficient number of amplitudes, append
    # amplitudes with "zeros" as necesary
    elif len(glbl.nuclear_basis['amplitudes']) < master.n_traj():
        dif = master.n_traj() - len(glbl.nuclear_basis['amplitudes'])
        glbl.nuclear_basis['amplitudes'].extend([0+0j for i in range(dif)])

    # finally -- update amplitudes in the bundle
    for i in range(master.n_traj()):
        master.traj[i].update_amplitude(glbl.nuclear_basis['amplitudes'][i])


def save_initial_wavefunction(master):
    """Sets the intial t=0 bundle in order to compute the autocorrelation
    function for subsequent time steps"""
    glbl.variables['bundle0'] = master.copy()
    # change the trajectory labels in this bundle to differentiate
    # them from trajctory labels in the master bundle. This avoids
    # cache collisions between trajetories in 'bundle0' and trajectories
    # in 'master'
    for i in range(glbl.variables['bundle0'].n_traj()):
        new_label = str(glbl.variables['bundle0'].traj[i].label)+'_0'
        glbl.variables['bundle0'].traj[i].label = new_label


def virtual_basis(master):
    """Add virtual basis funcions.

    If additional virtual basis functions requested, for each trajectory
    in bundle, add aditional basis functions on other states with zero
    amplitude.
    """
    for i in range(master.n_traj()):
        for j in range(glbl.propagate['n_states']):
            if j != master.traj[i].state():
                new_traj = master.traj[i].copy()
                new_traj.amplitude = 0j
                new_traj.set_state(j)
                master.add_trajectory(new_traj)


def make_origin_traj():
    """Construct a trajectory basis function at the origin
    specified in the input files"""
    ndim = len(glbl.nuclear_basis['geometries'][0])
    m_vec = np.array(glbl.nuclear_basis['masses'])
    w_vec = np.array(glbl.nuclear_basis['widths'])
    x_vec = np.array(glbl.nuclear_basis['geometries'][0])
    p_vec = np.array(glbl.nuclear_basis['momenta'][0])

    origin = trajectory.Trajectory(glbl.propagate['n_states'], ndim,
                                   width=w_vec, mass=m_vec, parent=0,
                                   kecoef=glbl.kecoef)

    origin.update_x(x_vec)
    origin.update_p(p_vec)
    origin.set_state(0)
    # if we need pes data to evaluate overlaps, determine that now
    if glbl.master_int.overlap_requires_pes:
        evaluate.update_pes_traj(origin)

    return origin
