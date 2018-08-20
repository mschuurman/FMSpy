""" Computes the Multiconfigurational Ehrenfest integrals 
for trajectories on multiple adiabatic electronic states

ref: Makhov, D. V.; Symonds, C.; Fernandez-Alberti, S.; Shalashilin, D. V. Chemical Physics 2017, 493, 200-218.
doi: https://doi.org/10.1016/j.chemphys.2017.04.003  
"""

import numpy as np
import nomad.compiled.nuclear_gaussian_ccs as nuclear

def elec_overlap(t1, t2):
	"""Returns < Phi | Phi' >, the electronic overlap integral ie. Kronecker delta"""
	if t1.state() == t2.state():
        return 1.
    else:
        return 0.

def traj_overlap(t1, t2, nuc_ovrlp=None):
    """Returns < Psi | Psi' >, the overlap integral of two trajectories.

    The bra and ket functions for the s_integral may be different
    (i.e. pseudospectral/collocation methods).
    """
    return s_integral(t1, t2, nuc_ovrlp)

def nuc_overlap(t1, t2):
    """ Returns < Chi | Chi' >, the nuclear overlap integral of two trajectories"""
    return nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                           t2.phase(),t2.widths(),t2.x(),t2.p())

def a_dot():
	"""This allows for the propagation of the electronic state weights a(t) for a single EBF."""
	pass

def s_integral():
	pass

def t_integral():
	pass

