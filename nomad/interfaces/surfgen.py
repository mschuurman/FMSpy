# file: surfgen.py
# 
# Routines for running surfgen.x surface evaluation
#
import sys
import os
import shutil
import copy
import pathlib
import numpy as np
import nomad.core.glbl as glbl
import nomad.core.surface as surface
import nomad.math.constants as constants
from ctypes import *

# surfgen library
libsurf = None

#---------------------------------------------------------------------
#
# Functions called from interface object
#
#---------------------------------------------------------------------
#
# init_interface: intialize surfgen and set up for evaluation
#
def init_interface():
    global libsurf
    err = 0

    # Check that $SURFGEN is set and load library, then check for input files.
    libsurfgen_path = os.environ['LIBSURFGEN']
    if not os.path.isfile(libsurfgen_path):
        print("Surfgen library not found: "+libsurfgen_path)
        sys.exit()
    libsurf = cdll.LoadLibrary(libsurfgen_path)

    err = check_surfgen_input('./input')
    if err != 0:
        print("Missing surfgen input files at: ./input")
        sys.exit()
        
    initialize_surfgen_potential()
        
#
# evaluate_trajectory: evaluate all reaqusted electronic structure
# information for a single trajectory
#
def evaluate_trajectory(traj, t=None):
    global libsurf

    n_atoms = int(traj.dim/3.)
    n_states = traj.nstates

    na = c_longlong(n_atoms)
    ns = c_longlong(n_states)

    na3 = 3 * n_atoms
    ns2 = n_states * n_states

    # convert to c_types for interfacing with surfgen shared
    # library.
    cgeom  = traj.x()
    energy = [0.0 for i in range(n_states)]
    cgrads = [0.0 for i in range(ns2*na3)]
    hmat   = [0.0 for i in range(ns2)]
    dgrads = [0.0 for i in range(ns2*na3)]

    cgeom = (c_double * na3)(*cgeom)
    energy= (c_double * n_states)(*energy)
    cgrads= (c_double * (ns2 * na3)) (*cgrads)
    hmat  = (c_double * ns2) (*hmat)
    dgrads= (c_double * (ns2 * na3)) (*dgrads)

    libsurf.evaluatesurfgen77_(byref(na), byref(ns), cgeom,
                               energy, cgrads, hmat, dgrads)
    cartgrd = np.array(np.reshape(cgrads,(na3,n_states,n_states),order='F'))
    potential = np.array(energy)

    # populate the surface object
    surf_gen = surface.Surface()
    surf_gen.add_data('geom', traj.x())
    surf_gen.add_data('potential', potential + glbl.properties['pot_shift'])

    # make sure the phase of Fij is consistent from one point to the next
    cgrad_phase = set_phase(traj, cartgrd)

    # add to the derivative object
    surf_gen.add_data('derivative', cgrad_phase)     

    return surf_gen

#
# evaluate_centroid: evaluate all requested electronic structure 
# information at a centroid
#
def evaluate_centroid(cent, t=None):

    return evaluate_trajectory(cent, t=None) 

#
# evaluate the coupling between electronic states
# 1. for adiabatic basis, this will just be the derivative coupling dotted into
#    the velocity
# 2. for diabatic basis, it will be the potential coupling, or something else
#
def evaluate_coupling(traj):
    """evaluate coupling between electronic states"""
    nstates = traj.nstates
    state   = traj.state

    # effective coupling is the nad projected onto velocity
    coup = np.zeros((nstates, nstates))
    vel  = traj.velocity()
    for i in range(nstates):
        if i != state:
            coup[state,i] = np.dot(vel, traj.derivative(state,i))
            coup[i,state] = -coup[state,i]
    traj.pes.add_data('coupling', coup)


#---------------------------------------------------------------------
#
# "Private" functions
#
#--------------------------------------------------------------------
#
# initialize_surfgen_potential: call initpotential_ to initialize
# surfgen surface evalutation.
#
def initialize_surfgen_potential():
    global libsurf

    os.chdir('./input')
    libsurf.initpotential_()
    os.chdir('../')

    return

#
# check_surfgen_input: check for all files necessary for successful
# surfgen surface evaluation.
#
def check_surfgen_input(path):
    # Input:
    #  path = path to directoy executing surfgen
    #
    # The following files are necessary for surfgen.x to run:
    #  hd.data, surfgen.in, coord.in, refgeom, irrep.in,
    #  error.log
    files = [path+'/hd.data', path+'/surfgen.in', path+'/coord.in',\
             path+'/refgeom', path+'/irrep.in']
    for i in range(len(files)):
        err = check_file_exists(files[i])
        if err != 0:
            print("File not found: "+files[i])

    return err

#
# check_file_exists: check if file exists
#
def check_file_exists(fname):
    err = 0
    if not os.path.isfile(fname):
        err = 1
    return err


#
# determine the phase of the computed coupling that yields smallest
# change from previous coupling
#
def set_phase(traj, new_grads):

    n_states = int(traj.nstates)

    # pull data to make consistent
    if traj.pes is not None:
        old_grads = np.transpose(np.array([[traj.derivative(i,j,geom_chk=False)
                                 for i in range(n_states)] for j in range(n_states)]), 
                                 axes=(2,0,1))
    else:
        old_grads = np.zeros((traj.dim, n_states, n_states))

    # ******** this needs to be amended to confirm the phase change in valid **************
    for i in range(n_states):
        for j in range(i):
            # if the previous coupling is vanishing, phase of new coupling is arbitrary
            if np.linalg.norm(old_grads[:,i,j]) > constants.fpzero:
                # check the difference between the vectors assuming phases of +1/-1
                norm_pos = np.linalg.norm( new_grads[:,i,j] - old_grads[:,i,j])
                norm_neg = np.linalg.norm(-new_grads[:,i,j] - old_grads[:,i,j])

                if norm_pos > norm_neg:
                    new_grads[:,i,j] *= -1.
                    new_grads[:,j,i] *= -1.

    return new_grads
