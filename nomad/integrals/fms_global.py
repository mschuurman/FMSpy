"""
Compute Bra-ket averaged Taylor expansion integrals over trajectories
traveling on adiabataic potentials
"""
import numpy as np
import nomad.compiled.nuclear_gaussian as nuclear

# Determines the Hamiltonian symmetry
hermitian = True

# Returns functional form of bra function ('dirac_delta', 'gaussian')
basis = 'gaussian'

def elec_overlap(t1, t2):
    """ Returns < Psi | Psi' >, the nuclear overlap integral of two trajectories"""
    if t1.state() == t2.state():
        return 1.
    else:
        return 0.


def nuc_overlap(t1, t2):
    """ Returns < Chi | Chi' >, the nuclear overlap integral of two trajectories"""
    return nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                           t2.phase(),t2.widths(),t2.x(),t2.p())


def traj_overlap(t1, t2, nuc_ovrlp=None):
    """Returns < Psi | Psi' >, the overlap integral of two trajectories.

    The bra and ket functions for the s_integral may be different
    (i.e. pseudospectral/collocation methods).
    """
    return s_integral(t1, t2, nuc_ovrlp)


def s_integral(t1, t2, nuc_ovrlp=None):
    """ Returns < Psi | Psi' >, the overlap of the nuclear
    component of the wave function only"""

    if nuc_ovrlp is None:
        nuc_ovrlp = nuc_overlap(t1, t2)

    return elec_overlap(t1, t2) * nuc_ovrlp


def t_integral(t1, t2, nuc_ovrlp=None):
    """Returns kinetic energy integral over trajectories."""
    if t1.state() != t2.state():
        return 0j

    else:
        if nuc_ovrlp is None:
            nuc_ovrlp = nuc_overlap(t1, t2)

        ke = nuclear.deld2x(nuc_ovrlp,t1.widths(),t1.x(),t1.p(),
                                      t2.widths(),t2.x(),t2.p())

        return -np.dot(ke, t1.kecoef)


def sdot_integral(t1, t2, nuc_ovrlp=None):
    """Returns the matrix element <Psi_1 | d/dt | Psi_2>."""
    if t1.state() != t2.state():
        return 0j

    else:
        if nuc_ovrlp is None:
            nuc_ovrlp = nuc_overlap(t1, t2)

        deldx = nuclear.deldx(nuc_ovrlp,t1.widths(),t1.x(),t1.p(),
                                        t2.widths(),t2.x(),t2.p())
        deldp = nuclear.deldp(nuc_ovrlp,t1.widths(),t1.x(),t1.p(),
                                        t2.widths(),t2.x(),t2.p())

        sdot = (np.dot(deldx,t2.velocity()) + np.dot(deldp,t2.force()) +
                1j * t2.phase_dot() * nuc_ovrlp)

        return sdot
