"""
Routines for running a Columbus computation.
"""
import sys
import os
import shutil
import subprocess
import numpy as np
import nomad.math.constants as constants
import nomad.core.glbl as glbl
import nomad.core.atom_lib as atom_lib
import nomad.core.trajectory as trajectory
import nomad.core.surface as surface
import nomad.integrals.centroid as centroid


# path to columbus executables
columbus_path = ''
# set to true if we want to compute electronic structure properties
comp_properties = True
# path to columbus input files
input_path = ''
# path to location of 'work'/'restart' directories
work_path = ''
# path to the location of restart files (i.e. mocoef files and civfl)
restart_path = ''
# atom labels
a_sym        = []
# atomic number
a_num        = []
# atomic masses (amu)
a_mass       = []
# by default, since this is the columbus module, assume atoms are
# 3-dimensional (i.e. Cartesian)
p_dim        = 3
# number of atoms
n_atoms      = 0
# total number of cartesian coordinates
n_cart       = 0
# number of drts (should usually be "1" for C1)
n_drt        = 1
# number of orbitals
n_orbs       = 0
# number of states in sate-averaged MCSCF
n_mcstates   = 0
# number of CI roots
n_cistates     = 0
# if DE between two states greater than de_thresh, ignore coupling
coup_de_thresh = 100.
# maximum angular momentum in basis set
max_l        = 1
# excitation level in CI
mrci_lvl     = 0
# amount of memory per process, in MB
mem_str      = ''


#----------------------------------------------------------------
#
# Functions called from interface object
#
#----------------------------------------------------------------
def init_interface():
    """Initializes the Columbus calculation from the Columbus input."""
    global columbus_path, input_path, work_path, restart_path, log_file
    global a_sym, a_num, a_mass, n_atoms, n_cart, p_dim, coup_de_thresh
    global n_orbs, n_mcstates, n_cistates, max_l, mrci_lvl, mem_str

    # set atomic symbol, number, mass,
    natm    = int(len(glbl.nuclear_basis['labels']) / p_dim)
    a_sym   = [glbl.nuclear_basis['labels'][p_dim*i] for i in range(natm)]

    a_data  = []
    # we need to go through this to pull out the atomic numbers for
    # correct writing of input
    for i in range(natm):
        if atom_lib.valid_atom(a_sym[i]):
            a_data.append(atom_lib.atom_data(a_sym[i]))
        else:
            raise ValueError('Atom: '+str(a_sym[i])+' not found in library')

    # masses are au -- columbus geom reads mass in amu
    a_mass  = [a_data[i][1]/constants.amu2au for i in range(natm)]
    a_num   = [a_data[i][2] for i in range(natm)]

    # confirm that we can see the COLUMBUS installation (pull the value
    # COLUMBUS environment variable)
    columbus_path = os.environ['COLUMBUS']
    if not os.path.isfile(columbus_path + '/ciudg.x'):
        raise FileNotFoundError('Cannot find COLUMBUS executables in: ' +
                                columbus_path)
    # ensure COLUMBUS input files are present locally
    if not os.path.exists('input'):
        raise FileNotFoundError('Cannot find COLUMBUS input files in: input')

    # setup working directories
    # input and restart are shared
    input_path    = glbl.home_path + '/input'
    restart_path  = glbl.home_path + '/restart'
    # ...but each process has it's own work directory
    work_path     = glbl.home_path + '/work.'+str(glbl.mpi['rank'])

    if os.path.exists(work_path):
        shutil.rmtree(work_path)
    os.makedirs(work_path)

    if glbl.mpi['rank'] == 0:
        if os.path.exists(restart_path):
            shutil.rmtree(restart_path)
        os.makedirs(restart_path)

    # copy input directory to home and copy file contents to work directory
    # we now asssume input directory is present in current directory
    for item in os.listdir('input'):
        local_file = os.path.join('input', item)

        work_file  = os.path.join(work_path, item)
        shutil.copy2(local_file, work_file)

    #    if glbl.mpi['rank'] == 0:
    #      input_file = os.path.join(input_path, item)
    #      shutil.copy2(local_file, input_file)

    # make sure process 0 is finished populating the input directory
    if glbl.mpi['parallel']:
        glbl.mpi['comm'].barrier()

    # now -- pull information from columbus input
    n_atoms    = natm
    n_cart     = natm * p_dim
    n_orbs     = int(read_pipe_keyword('input/cidrtmsin',
                                       'orbitals per irrep'))
    n_mcstates = int(read_nlist_keyword('input/mcscfin',
                                        'NAVST'))
    n_cistates = int(read_nlist_keyword('input/ciudgin.drt1',
                                        'NROOT'))
    mrci_lvl   = int(read_pipe_keyword('input/cidrtmsin',
                                       'maximum excitation level'))
    max_l      = ang_mom_dalton('input/daltaoin')

    # all COLUMBUS modules will be run with the amount of meomry specified by mem_per_core
    mem_str = str(int(glbl.iface_params['mem_per_core']))
    coup_de_thresh = float(glbl.iface_params['coup_de_thresh'])

    # Do some error checking to makes sure COLUMBUS calc is consistent with trajectory
    if n_cistates < int(glbl.propagate['n_states']):
        raise ValueError('n_cistates < n_states: t'+str(n_cistates)+' < '+str(glbl.propagate['n_states']))

    # generate one time input files for columbus calculations
    make_one_time_input()


def evaluate_trajectory(traj, t=None):
    """Computes MCSCF/MRCI energy and computes all couplings.

    For the columbus module, since gradients are not particularly
    time consuming, it's easier (and probably faster) to compute
    EVERYTHING at once (i.e. all energies, all gradients, all properties)
    Thus, if electronic structure information is not up2date, all methods
    call the same routine: run_single_point.
    """
    global n_cart

    label   = traj.label
    state   = traj.state()
    nstates = traj.nstates

    if label < 0:
        print('evaluate_trajectory called with ' +
              'id associated with centroid, label=' + str(label))

    # create surface object to hold potential information
    col_surf = surface.Surface()
    col_surf.add_data('geom', traj.x())

    # write geometry to file
    write_col_geom(traj.x())

    mo_restart, ci_restart = get_col_restart(traj)
    if not mo_restart:
        raise IOError('cannot find starting orbitals for mcscf')

    # generate integrals
    generate_integrals(label, t)

    # run mcscf
    run_col_mcscf(traj, t)
    col_surf.add_data('mo',pack_mocoef())

    # run mrci, if necessary
    potential, atom_pop = run_col_mrci(traj, ci_restart, t)
    col_surf.add_data('potential', potential + glbl.propagate['pot_shift'])
    col_surf.add_data('atom_pop', atom_pop)

    # run properties, dipoles, etc.
    [perm_dipoles, sec_moms] = run_col_multipole(traj)
    col_surf.add_data('sec_mom', sec_moms)
    dipoles = np.zeros((3,nstates,nstates), dtype=float)
    for i in range(nstates):
        dipoles[:,i,i] = perm_dipoles[:,i]

    # run transition dipoles
    init_states = [0, state]
    for i in init_states:
        for j in range(nstates):
            if i != j or (j in init_states and j < i):
                tr_dip = run_col_tdipole(label, i, j)
                dipoles[:,i,j] = tr_dip
                dipoles[:,j,i] = tr_dip
    col_surf.add_data('dipole',dipoles)

    # compute gradient on current state
    deriv = np.zeros((n_cart, nstates, nstates),dtype=float)
    grads = run_col_gradient(traj, t)
    deriv[:,state,state] = grads

    # run coupling to other states
    nad_coup = run_col_coupling(traj, potential, t)
    for i in range(nstates):
        if i != state:
            state_i = min(i,state)
            state_j = max(i,state)
            deriv[:, state_i, state_j] =  nad_coup[:, i]
            deriv[:, state_j, state_i] = -nad_coup[:, i]
    col_surf.add_data('derivative', deriv)

    # save restart files
    make_col_restart(traj)

    return col_surf


def evaluate_centroid(Cent, t=None):
    """Evaluates  all requested electronic structure information at a
    centroid."""
    global n_cart

    label   = Cent.label
    nstates = Cent.nstates

    if label >= 0:
        print('evaluate_centroid called with ' +
              'id associated with trajectory, label=' + str(label))

    state_i = min(Cent.states)
    state_j = max(Cent.states)

    # create surface object to hold potential information
    col_surf      = surface.Surface()

    col_surf.add_data('geom', Cent.x())

    # write geometry to file
    write_col_geom(Cent.x())

    mo_restart, ci_restart = get_col_restart(Cent)
    if not mo_restart:
        raise IOError('cannot find starting orbitals for mcscf')

    # generate integrals
    generate_integrals(label, t)

    # run mcscf
    run_col_mcscf(Cent, t)
    col_surf.add_data('mo',pack_mocoef())

    # run mrci, if necessary
    potential, atom_pop = run_col_mrci(Cent, ci_restart, t)
    col_surf.add_data('potential', potential + glbl.propagate['pot_shift'])
    col_surf.add_data('atom_pop', atom_pop)

    deriv = np.zeros((Cent.dim, nstates, nstates))
    if state_i != state_j:
        # run coupling to other states
        nad_coup = run_col_coupling(Cent, potential, t)
        deriv[:,state_i, state_j] =  nad_coup[:,state_j]
        deriv[:,state_j, state_i] = -nad_coup[:,state_j]

    col_surf.add_data('derivative', deriv)

    # save restart files
    make_col_restart(Cent)

    return col_surf

def evaluate_coupling(traj):
    """evaluate coupling between electronic states"""
    nstates = traj.nstates
    state   = traj.state()

    # effective coupling is the nad projected onto velocity
    coup = np.zeros((nstates, nstates),dtype='float')
    vel  = traj.velocity()
    for i in range(nstates):
        if i == state:
            continue
        coup[state,i] = np.dot(vel, traj.derivative(state,i))
        coup[i,state] = -coup[state,i]
    traj.pes.add_data('coupling', coup)
    return


#----------------------------------------------------------------
#
# Routines for running columbus
#
#---------------------------------------------------------------
def make_one_time_input():
    """Creates a Columbus input for MRCI calculations."""
    global mem_str
    global columbus_path, work_path

    # all calculations take place in work_dir
    os.chdir(work_path)

    # rotation matrix
    with open('rotmax', 'w') as rfile:
        rfile.write('  1  0  0\n  0  1  0\n  0  0  1')

    # cidrtfil files
    with open('cidrtmsls', 'w') as cidrtmsls, open('cidrtmsin', 'r') as cidrtmsin:
        run_prog('init', 'cidrtms.x', args=['-m',mem_str],
                                   in_pipe=cidrtmsin,
                                   out_pipe=cidrtmsls)
    shutil.move('cidrtfl.1', 'cidrtfl.ci')

    with open('cidrtmsls.cigrd', 'w') as cidrtmsls_grd, \
         open('cidrtmsin.cigrd', 'r') as cidrtmsin_grd:
        run_prog('init', 'cidrtms.x', args=['-m',mem_str],
                                   in_pipe=cidrtmsin_grd,
                                   out_pipe=cidrtmsls_grd)
    shutil.move('cidrtfl.1', 'cidrtfl.cigrd')

    # check if hermitin exists, if not, copy daltcomm
    if not os.path.exists('hermitin'):
        shutil.copy('daltcomm', 'hermitin')

    # make sure ciudgin file exists
    shutil.copy('ciudgin.drt1', 'ciudgin')


def generate_integrals(label, t):
    """Runs Dalton to generate AO integrals."""
    global work_path

    os.chdir(work_path)

    # run unik.gets.x script
    with open('unikls', 'w') as unikls:
        run_prog(label, 'unik.gets.x', out_pipe=unikls)

    # run hernew
    run_prog(label, 'hernew.x')
    shutil.move('daltaoin.new', 'daltaoin')

    # run dalton.x
    shutil.copy('hermitin', 'daltcomm')

    with open('hermitls', 'w') as hermitls:
        run_prog(label, 'dalton.x', args=['-m', mem_str],
                             out_pipe = hermitls)

    append_log(label,'integral', t)


def run_col_mcscf(traj, t):
    """Runs MCSCF program."""
    global n_mcstates, n_drt, mrci_lvl, mem_str
    global work_path

    label = traj.label

    if type(traj) is trajectory.Trajectory:
        state = traj.state()
    else:
        state = min(traj.states)

    os.chdir(work_path)

    # allow for multiple DRTs in the mcscf part. For example, We may
    # want to average over a' and a" in a tri-atomic case
    for i in range(1, n_drt+1):
        shutil.copy('mcdrtin.' + str(i), 'mcdrtin')
        with open('mcdrtls', 'w') as mcdrtls, open('mcdrtin', 'r') as mcdrtin:
            run_prog(label, 'mcdrt.x', args     = ['-m', mem_str],
                                       in_pipe  = mcdrtin,
                                       out_pipe = mcdrtls)

        with open('mcuftls', 'w') as mcuftls:
            run_prog(label, 'mcuft.x', out_pipe = mcuftls)

        # save formula tape and log files for each DRT
        shutil.copy('mcdrtfl', 'mcdrtfl.' + str(i))
        shutil.copy('mcdftfl', 'mcdftfl.' + str(i))
        shutil.copy('mcuftls', 'mcuftls.' + str(i))
        shutil.copy('mcoftfl', 'mcoftfl.' + str(i))

    # if running cas dynamics (i.e. no mrci), make sure we compute the
    # mcscf/cas density (for gradients and couplings)
    if mrci_lvl == 0:
        with open('mcdenin', 'w', encoding='utf-8') as mcden:
            mcden.write('MCSCF')
            # diagonal densities (for gradients)
            for i in range(n_mcstates):
                mcden.write('1  {:2d}  1  {:2d}').format(i, i)
            # off-diagonal densities (for couplings)
            for i in range(n_mcstates):
                mcden.write('1  {:2d}  1  {:2d}').format(min(i, state),
                                                         max(i, state))

    # try running mcscf a couple times this can be tweaked if one
    # develops other strategies to deal with convergence problems
    converged = False
    run_max   = 3
    n_run     = 0
    while not converged and n_run < run_max:
        n_run += 1
        if n_run == 3:
            # disable orbital-state coupling if convergence an issue
            ncoupl = int(read_nlist_keyword('mcscfin', 'ncoupl'))
            niter  = int(read_nlist_keyword('mcscfin', 'niter'))
            set_nlist_keyword('mcscfin', 'ncoupl', niter+1)

        run_prog(label, 'mcscf.x', args=['-m', mem_str])

        # check convergence
        with open('mcscfls', 'r') as ofile:
            for line in ofile:
                if '*converged*' in line:
                    converged = True
                    break

    # if not converged, we have to die here...
    if not converged:
        raise TimeoutError('MCSCF not converged.')

    # save output
    shutil.copy('mocoef_mc', 'mocoef')

    # grab mcscfls output
    append_log(label,'mcscf', t)


def run_col_mrci(traj, ci_restart, t):
    """Runs MRCI if running at that level of theory."""
    global n_atoms, n_cistates, max_l, mem_str
    global work_path

    os.chdir(work_path)
    label = traj.label

    # get a fresh ciudgin file
    shutil.copy(input_path + '/ciudgin.drt1', 'ciudgin')

    # if old ci vectors are present, set NOLDV=n_cistatese
    if ci_restart:
        set_nlist_keyword('ciudgin','NOLDV', n_cistates)
        set_nlist_keyword('ciudgin','NBKITR', 0)

    # determine if trajectory or centroid, and compute densities
    # accordingly
    if type(traj) is trajectory.Trajectory:

        # perform density transformation for gradient computations
        int_trans = True
        # compute densities between all states and trajectory state
        tran_den = []
        init_states = [0, traj.state()]
        for i in init_states:
            for j in range(traj.nstates):
                if i != j and not (j in init_states and j < i):
                    tran_den.append([min(i,j)+1, max(i,j)+1])

    # else, this is a centroid
    else:
        # only need gradient if statei != statej
        state_i = min(traj.states)
        state_j = max(traj.states)
        int_trans = (traj.states[0] != traj.states[1])
        tran_den  = [[state_i+1, state_j+1]]

    # append entries in tran_den to ciudgin file
    with open('ciudgin', 'r') as ciudgin:
        ci_file = ciudgin.readlines()
    with open('ciudgin', 'w') as ciudgin:
        for line in ci_file:
            ciudgin.write(line)
            if '&end' in line:
                break
        ciudgin.write('transition\n')
        for i in range(len(tran_den)):
            ciudgin.write('  1 {:2d}  1 {:2d}\n'.format(tran_den[i][0],
                                                        tran_den[i][1]))

    # make sure we point to the correct formula tape file
    link_force('cidrtfl.ci', 'cidrtfl')
    link_force('cidrtfl.ci', 'cidrtfl.1')

    # perform the integral transformation
    with open('tranin', 'w') as ofile:
        ofile.write('&input\nLUMORB=0\n&end')
    run_prog(label, 'tran.x', args=['-m', mem_str])

    # run mrci
    run_prog(label, 'ciudg.x', args=['-m', mem_str])

    ci_ener = []
    ci_res  = []
    ci_tol  = []
    mrci_iter = False
    converged = True
    sys.stdout.flush()
    with open('ciudgsm', 'r') as ofile:
        for line in ofile:
            if 'beginning the ci' in line:
                mrci_iter = True
            if 'final mr-sdci  convergence information' in line and mrci_iter:
                for i in range(n_cistates):
                    ci_info = ofile.readline().split()
                    try:
                        ci_info.remove('#') # necessary due to unfortunate columbus formatting
                    except ValueError:
                        pass
                    ci_ener.append(float(ci_info[3]))
                    ci_res.append(float(ci_info[6]))
                    ci_tol.append(float(ci_info[7]))
                    converged = converged and ci_res[-1] <= ci_tol[-1]
                break

    # determine convergence...
    if not converged:
        raise TimeoutError('MRCI did not converge for trajectory ' + str(label))

    # if we're good, update energy array
    energies = np.array([ci_ener[i] for i in range(traj.nstates)],dtype=float)

    # now update atom_pops
    ist = -1
    atom_pops = np.zeros((n_atoms, traj.nstates))
    with open('ciudgls', 'r') as ciudgls:
        for line in ciudgls:
            if '   gross atomic populations' in line:
                ist += 1
                # only get populations for lowest traj.nstates states
                if ist == traj.nstates:
                    break
                pops = []
                for i in range(int(np.ceil(n_atoms/6.))):
                    for j in range(max_l+3):
                        nxtline = ciudgls.readline()
                        if 'total' in line:
                            break
                    l_arr = nxtline.split()
                    pops.extend(l_arr[1:])
                atom_pops[:, ist] = np.array(pops, dtype=float)

    # grab mrci output
    append_log(label,'mrci', t)

    # transform integrals using cidrtfl.cigrd
    if int_trans:
        frzn_core = int(read_nlist_keyword('cigrdin', 'assume_fc'))
        if frzn_core == 1:
            os.remove('moints')
            os.remove('cidrtfl')
            os.remove('cidrtfl.1')
            link_force('cidrtfl.cigrd', 'cidrtfl')
            link_force('cidrtfl.cigrd', 'cidrtfl.1')
            shutil.copy(input_path + '/tranin', 'tranin')
            run_prog(label, 'tran.x', args=['-m', mem_str])

    return energies, atom_pops


def run_col_multipole(traj):
    """Runs dipoles / second moments."""
    global p_dim, mrci_lvl, mem_str
    global work_path

    os.chdir(work_path)

    nst       = traj.nstates
    dip_moms  = np.zeros((p_dim, traj.nstates))
    sec_moms  = np.zeros((p_dim, p_dim, traj.nstates))

    if mrci_lvl == 0:
        type_str   = 'mc'
    else:
        type_str   = 'ci'

    for istate in range(nst):
        i1 = istate + 1
        link_force('nocoef_' + str(type_str) + '.drt1.state' + str(i1),
                   'mocoef_prop')
        run_prog(traj.label, 'exptvl.x', args=['-m', mem_str])

        with open('propls', 'r') as prop_file:
            for line in prop_file:
                if 'Dipole moments' in line:
                    for j in range(5):
                        line = prop_file.readline()
                    l_arr = line.split()
                    dip_moms[:,istate] = np.array([float(l_arr[1]),
                                                   float(l_arr[2]),
                                                   float(l_arr[3])])
                if 'Second moments' in line:
                    for j in range(5):
                        line = prop_file.readline()
                    l_arr = line.split()
                    for j in range(5):
                        line = prop_file.readline()
                    l_arr.extend(line.split())
                    # NOTE: we're only taking the diagonal elements
                    inds = [1,2,3,4,6,7]
                    raw_dat = np.array([float(l_arr[j]) for j in inds])
                    map_arr = [[0,1,2],[1,3,4],[2,4,5]]
                    for i in range(p_dim):
                        for j in range(i+1):
                            sec_moms[i,j,istate] = raw_dat[map_arr[i][j]]
                            sec_moms[j,i,istate] = raw_dat[map_arr[j][i]]

        os.remove('mocoef_prop')

    return dip_moms, sec_moms


def run_col_tdipole(label, state_i, state_j):
    """Computes transition dipoles between ground and excited state,
    and between trajectory states and other state."""
    global p_dim, mrci_lvl, mem_str
    global work_path

    os.chdir(work_path)

    # make sure we point to the correct formula tape file
    link_force('civfl', 'civfl.drt1')
    link_force('civout', 'civout.drt1')
    link_force('cirefv', 'cirefv.drt1')

    i1 = min(state_i, state_j) + 1
    j1 = max(state_i, state_j) + 1

    if state_i == state_j:
        return None

    if mrci_lvl == 0:
        with open('transftin', 'w') as ofile:
            ofile.write('y\n1\n' + str(j1) + '\n1\n' + str(i1))
        run_prog(label, 'transft.x', in_pipe='transftin', out_pipe='transftls')

        with open('transmomin', 'w') as ofile:
            ofile.write('MCSCF\n1 ' + str(j1) + '\n1\n' + str(i1))
        run_prog(label, 'transmom.x', args=['-m', mem_str])

        os.remove('mcoftfl')
        shutil.copy('mcoftfl.1', 'mcoftfl')

    else:
        with open('trnciin', 'w') as ofile:
            ofile.write(' &input\n lvlprt=1,\n nroot1=' + str(i1) + ',\n' +
                        ' nroot2=' + str(j1) + ',\n drt1=1,\n drt2=1,\n &end')
        run_prog(label, 'transci.x', args=['-m', mem_str])

        shutil.move('cid1trfl', 'cid1trfl.' + str(i1) + '.' + str(j1))

    tran_dip = np.zeros(p_dim)

    with open('trncils', 'r') as trncils:
        for line in trncils:
            if 'total (elec)' in line:
                line_arr = line.split()
                for dim in range(p_dim):
                    tran_dip[dim] = float(line_arr[dim+2])
                    tran_dip[dim] = float(line_arr[dim+2])

    return tran_dip


def run_col_gradient(traj, t):
    """Performs integral transformation and determine gradient on
    trajectory state."""
    global mrci_lvl, mem_str
    global work_path

    os.chdir(work_path)
    shutil.copy(input_path + '/cigrdin', 'cigrdin')
    tstate = traj.state() + 1

    if mrci_lvl > 0:
        link_force('cid1fl.drt1.state' + str(tstate), 'cid1fl')
        link_force('cid2fl.drt1.state' + str(tstate), 'cid2fl')
        shutil.copy(input_path + '/trancidenin', 'tranin')
    else:
        link_force('mcsd1fl.' + str(tstate), 'cid1fl')
        link_force('mcsd2fl.' + str(tstate), 'cid2fl')
        set_nlist_keyword('cigrdin', 'samcflag', 1)
        shutil.copy(input_path + '/tranmcdenin', 'tranin')

    # run cigrd
    set_nlist_keyword('cigrdin', 'nadcalc', 0)
    run_prog(traj.label, 'cigrd.x', args=['-m', mem_str])

    os.remove('cid1fl')
    os.remove('cid2fl')
    shutil.move('effd1fl', 'modens')
    shutil.move('effd2fl', 'modens2')

    # run tran
    run_prog(traj.label, 'tran.x', args=['-m', mem_str])

    os.remove('modens')
    os.remove('modens2')

    # run dalton
    shutil.copy(input_path + '/abacusin', 'daltcomm')
    with open('abacusls', 'w') as abacusls:
        run_prog(traj.label, 'dalton.x', args=['-m', mem_str],
                                         out_pipe=abacusls)

    shutil.move('abacusls', 'abacusls.grad')

    with open('cartgrd', 'r') as cartgrd:
        lines = cartgrd.readlines()
    grad     = [lines[i].split() for i in range(len(lines))]
    gradient = np.array([item.replace('D', 'e') for row in grad
                             for item in row], dtype=float)

    shutil.move('cartgrd', 'cartgrd.s'+str(traj.state())+'.'+str(traj.label))

    # grab cigrdls output
    append_log(traj.label,'cigrd', t)

    return gradient


def run_col_coupling(traj, ci_ener, t):
    """Computes couplings to states within prescribed DE window."""
    global n_cart, coup_de_thresh, mrci_lvl, mem_str
    global input_path, work_path

    if type(traj) is trajectory.Trajectory:
        t_state    = traj.state()
        c_states   = range(traj.nstates)
        delta_e_max = coup_de_thresh
    elif type(traj) is centroid.Centroid:
        t_state    = min(traj.states)
        c_states   = [max(traj.states)]
        # if computing coupling regardless of delta e,
        # set threshold to something we know won't trigger
        # the ignoring of the coupling
        delta_e_max = 2.*(ci_ener[-1] - ci_ener[0])

    nad_coupl = np.zeros((n_cart, traj.nstates))

    os.chdir(work_path)

    # copy some clean files to the work directory
    shutil.copy(input_path + '/cigrdin', 'cigrdin')
    set_nlist_keyword('cigrdin', 'nadcalc', 1)
    if mrci_lvl == 0:
        set_nlist_keyword('cigrdin', 'samcflag', 1)
        shutil.copy(input_path + '/tranmcdenin', 'tranin')
    else:
        shutil.copy(input_path + '/trancidenin', 'tranin')

    shutil.copy(input_path + '/abacusin', 'daltcomm')
    insert_dalton_key('daltcomm', 'COLBUS', '.NONUCG')

    # loop over states to compute coupling to
    for c_state in c_states:
        if c_state == t_state or abs(ci_ener[c_state] -
                                     ci_ener[t_state]) > delta_e_max:
            continue

        s1 = str(min(t_state, c_state) + 1).strip()
        s2 = str(max(t_state, c_state) + 1).strip()

        if mrci_lvl == 0:
            link_force('mcsd1fl.trd' + s1 + 'to' + s2, 'cid1fl.tr')
            link_force('mcsd2fl.trd' + s1 + 'to' + s2, 'cid2fl.tr')
            link_force('mcad1fl.' + s1 + s2, 'cid1trfl')
        else:
            link_force('cid1fl.trd' + s1 + 'to' + s2, 'cid1fl.tr')
            link_force('cid2fl.trd' + s1 + 'to' + s2, 'cid2fl.tr')
            link_force('cid1trfl.' + s1 + '.' + s2, 'cid1trfl')

        set_nlist_keyword('cigrdin', 'drt1', 1)
        set_nlist_keyword('cigrdin', 'drt2', 1)
        set_nlist_keyword('cigrdin', 'root1', s1)
        set_nlist_keyword('cigrdin', 'root2', s2)

        run_prog(traj.label, 'cigrd.x', args=['-m', mem_str])

        shutil.move('effd1fl', 'modens')
        shutil.move('effd2fl', 'modens2')

        run_prog(traj.label, 'tran.x', args=['-m', mem_str])

        with open('abacusls', 'w') as abacusls:
            run_prog(traj.label, 'dalton.x', args=['-m', mem_str],
                                             out_pipe=abacusls)

        # read in cartesian gradient and save to array
        with open('cartgrd', 'r') as cartgrd:
            lines = cartgrd.read().splitlines()
        grad = [lines[i].split() for i in range(len(lines))]
        coup_vec = np.array([item.replace('D', 'e') for row in grad
                             for item in row], dtype=float)

        delta_e = ci_ener[c_state] - ci_ener[t_state]
        nad_coupl[:,c_state] = coup_vec / delta_e
        shutil.move('cartgrd', 'cartgrd.nad.' + str(s1) + '.' + str(s2))

        # grab mcscfls output
        append_log(traj.label,'nad', t)

    # set the phase of the new coupling vectors using the cached data
    nad_coupl_phased = get_adiabatic_phase(traj, nad_coupl)

    return nad_coupl_phased


def make_col_restart(traj):
    """Saves mocoef and ci files to restart directory."""
    global restart_path, work_path

    os.chdir(work_path)
    label = traj.label

    # move orbitals
    shutil.move(work_path+'/mocoef', restart_path+'/mocoef.'+str(label))

    # move all ci vector, ci info files
    # need to investigate behavior of ciudg with respect restarts and IO
    # in  fortran, symlink to ci vector file is destroyed, replaced with
    # new file. Here, the ci vector is seemingly edited, meaning that when
    # ciudg finishes, and symlink remains and points to an edited file.
    # In this case, one simply removes the symlink, no need to edit file
    # in restart directory.
    if os.path.islink(work_path+'/civfl'):
        os.unlink(work_path+'/civfl')
    else:
        shutil.move(work_path+'/civfl',  restart_path+'/civfl.'+str(label))

    if os.path.islink(work_path+'/civout'):
        os.unlink(work_path+'/civout')
    else:
        shutil.move(work_path+'/civout', restart_path+'/civout.'+str(label))

    if os.path.islink(work_path+'/cirefv'):
        os.unlink(work_path+'/cirefv')
    else:
        shutil.move(work_path+'/cirefv', restart_path+'/cirefv.'+str(label))

    # do some cleanup
    if os.path.isfile('cirdrtfl'):   os.unlink('cidrtfl')
    if os.path.isfile('cirdrtfl.1'): os.unlink('cidrtfl.1')
    if os.path.isfile('aoints'):     os.unlink('aoints')
    if os.path.isfile('aoints2'):    os.unlink('aoints2')
    if os.path.isfile('modens'):     os.unlink('modens')
    if os.path.isfile('modens2'):    os.unlink('modens2')
    if os.path.isfile('cid1fl.tr'):  os.unlink('cid1fl.tr')
    if os.path.isfile('cid2fl.tr'):  os.unlink('cid2fl.tr')
    if os.path.isfile('cid1trfl'):   os.unlink('cid1trfl')
    if os.path.isfile('civfl.drt1'): os.unlink('civfl.drt1')
    if os.path.isfile('civout.drt1'):os.unlink('civout.drt1')
    if os.path.isfile('cirefv.drt1'):os.unlink('cirefv.drt1')


def get_col_restart(traj):
    """Get restart mocoef file and ci vectors for columbus calculation.

    1. failure to find mocoef file is fatal.
    2. failure to find ci files is OK

    MOCOEF
    1. If first step and parent-less trajectory, take what's in input.
    2. If first step of spawned trajectory, take parents restart info.
    3. If first step of centroid, take one of parent's restart info.

    CIUDG
    1. Copys/links CI restart files to working directory.
    2. If no ci vectors, simply start CI process from scratch
    """
    global work_path, restart_path

    os.chdir(work_path)
    mocoef_file = restart_path + '/mocoef.'
    lbl_str = str(traj.label) # string for trajectory label
    par_str = ''              # string for parent trajectory label

    if type(traj) is centroid.Centroid:
        # centroids have two parents
        par_arr = [str(traj.parents[i]) for i in range(len(traj.parents))]
    else:
        # if trajectory, there is a single parent
        par_arr = [str(traj.parent)]


    mo_restart = False
    ci_restart = False

    # MOCOEF RESTART FILES
    # if we have some orbitals in memory, write those out
    if traj.pes is not None and 'mo' in traj.pes.avail_data():
        write_mocoef('mocoef', traj.pes.get_data('mo'))
        mo_restart = True
    # if restart file exists, create symbolic link to it
    elif os.path.exists(mocoef_file+lbl_str):
        shutil.copy(mocoef_file+lbl_str, 'mocoef')
        mo_restart = True

    # if we still haven't found an mocoef file, check restart files
    # of parents [relevant if we've just spawned and this is first
    # pes evaluation for the child
    if not mo_restart:
        print('looking for parent restart...')
        for i in range(len(par_arr)):
            print('checking: '+mocoef_file+par_arr[i])
            if os.path.exists(mocoef_file+par_arr[i]):
                shutil.copy(mocoef_file+par_arr[i], 'mocoef')
                mo_restart = True
                print('found: '+mocoef_file+par_arr[i])
                par_str = par_arr[i]
                break
        sys.stdout.flush()

    if not mo_restart:
        # else, just take the mocoef file we have lying around
        if os.path.exists(work_path+'/mocoef'):
            mo_restart = True
        # else, we're out of luck
        else:
            mo_restart = False

    # CI RESTART FILES
    # if restart file exists, create symbolic link to it
    civfl  = restart_path + '/civfl.' + lbl_str
    civout = restart_path + '/civout.' + lbl_str
    cirefv = restart_path + '/cirefv.' + lbl_str

    civfl_p  = restart_path + '/civfl.' + par_str
    civout_p = restart_path + '/civout.' + par_str
    cirefv_p = restart_path + '/cirefv.' + par_str

    # if restart file exists, create symbolic link to it
    if (os.path.isfile(civfl) and os.path.isfile(civout)
            and os.path.isfile(cirefv)):
        ci_restart = True
    # if parent restart files exists, create symbolic link to it
    elif (os.path.isfile(civfl_p) and os.path.isfile(civout_p)
            and os.path.isfile(cirefv_p)):
        shutil.copy(civfl_p, civfl)
        shutil.copy(civout_p, civout)
        shutil.copy(cirefv_p, cirefv)
        ci_restart = True
    # else no ci restart
    else:
        ci_restart = False

    if ci_restart:
        link_force(civfl, work_path+'/civfl')
        link_force(civout, work_path+'/civout')
        link_force(cirefv, work_path+'/cirefv')

    return mo_restart, ci_restart


def get_adiabatic_phase(traj, new_coup):
    """Determines the phase of the computed coupling that yields smallest
    change from previous coupling."""
    global n_cart

    label = traj.label
    if type(traj) is trajectory.Trajectory:
        state = traj.state()
    else:
        state = min(traj.states)

    # pull data to make consistent
    if traj.pes is not None:
        old_coup = np.transpose(
                   np.array([traj.derivative(min(state,i),max(state,i),geom_chk=False)
                                             for i in range(traj.nstates)]))
    else:
        old_coup = np.zeros((n_cart, traj.nstates))

    for i in range(traj.nstates):
        # if the previous coupling is vanishing, phase of new coupling is arbitrary
        if np.linalg.norm(old_coup[:,i]) > constants.fpzero:
            # check the difference between the vectors assuming phases of +1/-1
            norm_pos = np.linalg.norm( new_coup[:,i] - old_coup[:,i])
            norm_neg = np.linalg.norm(-new_coup[:,i] - old_coup[:,i])

            if norm_pos > norm_neg:
                new_coup[:,i] *= -1.

    return new_coup


#-----------------------------------------------------------------
#
# File parsing
#
#-----------------------------------------------------------------
def run_prog(tid, prog_name, args=None, in_pipe=None, out_pipe=None):
    """Tries to run a Columbus program executable. If error is
    raised, return False, else True"""

    arg    = [str(prog_name)]
    kwargs = dict()

    # first argument is executable, plus any arguments passed to executable
    if args:
        arg.extend(args)

    # if we need to pipe input
    if in_pipe:
        kwargs['stdin'] = in_pipe

    # if we need to pipe output
    if out_pipe:
        kwargs['stdout'] = out_pipe

    # append check for error code
    kwargs['check'] = True
    kwargs['universal_newlines'] = True

    subprocess.run(arg, **kwargs)
    # if got here, return code not caught as non-zero, but check
    # bummer file to be sure error code not caught by Columbus
    if not prog_status():
        raise TimeoutError(str(prog_name)+' returned error, traj='+str(tid))


def prog_status():
    """Opens bummer file, checks to see if fatal error message
    has been written. If so, return False, else, return True"""

    try:
        with open("bummer", "r") as f:
            bummer = f.readlines()

    except EnvironmentError:
        # if bummer not here, return True
        return True

    bstr = "".join(bummer)

    return bstr.find('fatal') == -1 or bstr.find('nonfatal') != -1


def append_log(label, listing_file, time):
    """Grabs key output from columbus listing files.

    Useful for diagnosing electronic structure problems.
    """
    # check to see if time is given, if not -- this is a spawning
    # situation
    if time is None:
        tstr = "spawning"
    else:
        tstr = str(time)

    # open the running log for this process
    log_file = open(glbl.home_path+'/columbus.log.'+str(glbl.mpi['rank']), 'a')

    log_file.write(" time="+tstr+" trajectory="+str(label)+
                   ": "+str(listing_file)+" summary -------------\n")

    if listing_file == 'integral':
        with open('hermitls', 'r') as hermitls:
            for line in hermitls:
                if 'Bond distances' in line:
                    while 'Nuclear repulsion energy' not in line:
                        log_file.write(line)
                        line = hermitls.readline()
                    break
    elif listing_file == 'mcscf':
        with open('mcscfls', 'r') as mcscfls:
            for line in mcscfls:
                if 'final mcscf' in line:
                    while len(line.rstrip()) != 0:
                        log_file.write(line)
                        line = mcscfls.readline()
                    break
    elif listing_file == 'mrci':
        with open('ciudgsm', 'r') as ciudgls:
            ci_iter = False
            for line in ciudgls:
                if 'beginning the ci iterative':
                    ci_iter = True
                if 'final mr-sdci  convergence information' in line and ci_iter:
                    while len(line.rstrip()) != 0:
                        log_file.write(line)
                        line = ciudgls.readline()
                    break
    elif listing_file == 'cigrd':
        with open('cigrdls', 'r') as cigrdls:
            for line in cigrdls:
                if 'RESULTS' in line:
                    while 'effective' not in line:
                        log_file.write(line)
                        line = cigrdls.readline()
                    break
    elif listing_file == 'nad':
        with open('cigrdls', 'r') as cigrdls_nad:
            for line in cigrdls_nad:
                if 'RESULTS' in line:
                    while 'effective' not in line:
                        log_file.write(line)
                        line = cigrdls_nad.readline()
                    break
    else:
        print('listing file: ' + str(listing_file) + ' not recognized.')

    log_file.close()


def write_col_geom(geom):
    """Writes a array of atoms to a COLUMBUS style geom file."""
    global n_atoms, p_dim, a_sym, a_num, a_mass
    global work_path

    os.chdir(work_path)

    f = open('geom', 'w', encoding='utf-8')
    for i in range(n_atoms):
        f.write(' {:2s}   {:3.1f}  {:12.8f}  {:12.8f}  {:12.8f}  {:12.8f}'
                '\n'.format(a_sym[i], a_num[i],
                            geom[p_dim*i],geom[p_dim*i+1],geom[p_dim*i+2],
                            a_mass[i]))
    f.close()


def read_pipe_keyword(infile, keyword):
    """Reads from a direct input file via keyword search."""
    f = open(infile, 'r', encoding='utf-8')
    for line in f:
        if keyword in line:
            f.close()
            return line.split()[0]


def read_nlist_keyword(infile, keyword):
    """Reads from a namelist style input."""
    f = open(infile, 'r', encoding='utf-8')
    for line in f:
        if keyword in line:
            f.close()
            line = line.rstrip("\r\n")
            return line.split('=', 1)[1].strip(' ,')


def set_nlist_keyword(file_name, keyword, value):
    """Writes a namelist style input."""
    outfile = str(file_name) + '.tmp'
    key_found = False
    with open(file_name, 'r') as ifile, open(outfile, 'w') as ofile:
        for line in ifile:
            if keyword in line:
                ofile.write(str(keyword) + ' = ' + str(value) + '\n')
                key_found = True
            elif '&end' in line and not key_found:
                ofile.write(str(keyword) + ' = ' + str(value) + ',\n')
                ofile.write(line)
            else:
                ofile.write(line)
    shutil.move(outfile, file_name)


def insert_dalton_key(infile, keyword, value):
    """Insert a Dalton keyword.

    This is a pretty specialized function given the idiosyncracies of
    dalton input. The keyword must already exist in the file.
    """
    with open(infile, 'r') as ifile, open('tempfile', 'w') as ofile:
        for line in ifile:
            ofile.write(line)
            if keyword in line:
                ofile.write(value + '\n')
    shutil.move('tempfile', infile)


def ang_mom_dalton(infile):
    """Finds maximum ang. mom. in basis set from dalton."""
    max_l = 0
    with open(infile, 'r') as daltaoin:
        for i in range(4):
            line = daltaoin.readline()
        l_arr = line.split()
        n_grps = int(l_arr[1])
        for i in range(n_grps):
            line = daltaoin.readline()
            l_arr = line.split()
            n_atm = int(l_arr[1])
            n_ang = int(l_arr[2]) - 1
            # max_l on first line
            max_l = max(max_l, n_ang)
            n_con = [int(l_arr[j]) for j in range(3, 3+n_ang+1)]
            for j in range(n_atm):
                line = daltaoin.readline()
            for j in range(len(n_con)):
                for k in range(n_con[j]):
                    line = daltaoin.readline()
                    nprim  = int(line.split()[1])
                    n_line = int(np.ceil(float(line.split()[2])/3.))
                    for l in range(nprim * n_line):
                        line = daltaoin.readline()
    return max_l


def file_len(fname):
    """Returns the number of lines in a file."""
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def link_force(target, link_name):
    """Creates a symbolic link, overwriting existing link if necessary."""
    try:
        os.symlink(target, link_name)
    except FileExistsError:
        os.unlink(link_name)
        os.symlink(target, link_name)


def pack_mocoef():
    """Loads orbitals from a mocoef file."""
    mos = np.loadtxt('mocoef',dtype=bytes,delimiter='\n').astype(str)
    return mos


def write_mocoef(fname, mos):
    """Writes orbitals to mocoef file."""
    np.savetxt(str(fname), mos, fmt="%s")

