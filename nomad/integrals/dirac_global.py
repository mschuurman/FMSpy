"""
Computes matrix elements over a dirac delta function at the centre of
trajectory 1 and trajectory 2.
"""
import numpy as np
import nomad.core.glbl as glbl
import nomad.compiled.nuclear_dirac as dirac
import nomad.compiled.nuclear_gaussian as gauss


# Let propagator know if we need data at centroids to propagate
require_centroids = False

# Determines the Hamiltonian symmetry
hermitian = False

# functional form of bra function ('dirac_delta', 'gaussian')
basis = 'dirac_delta'

def elec_overlap(t1, t2):
    """ Returns < Psi | Psi' >, the nuclear overlap integral of two trajectories"""
    if t1.state() == t2.state():
        return 1.
    else:
        return 0.

def traj_overlap(traj1, traj2, nuc_only=False):
    """ Returns < Psi | Psi' >, the overlap integral of two trajectories"""
    if traj1.state() != traj2.state() and not nuc_only:
        return 0j
    else:
        return gauss.overlap(traj1.phase(),traj1.widths(),traj1.x(),traj1.p(),
                             traj2.phase(),traj2.widths(),traj2.x(),traj2.p())


def s_integral(traj1, traj2, nuc_only=False, nuc_ovrlp=None):
    """ Returns < chi | Psi' >, the overlap integral under the pseudospectral
    projection."""
    if traj1.state() != traj2.state() and not nuc_only:
        return 0j
    else:
        if nuc_ovrlp is None:
            return dirac.overlap(traj1.x(),
                                 traj2.phase(),traj2.widths(),traj2.x(),traj2.p())
        else:
            return nuc_ovrlp

def v_integral(traj1, traj2, centroid=None, nuc_ovrlp=None):
    """ Returns < delta(R-R1) | V | g2 > if state1 = state2, else
    returns < delta(R-R1) | F . d/dR | g2 > """
    if nuc_ovrlp is None:
        nuc_ovrlp = dirac.overlap(traj1.x(),
                             traj2.phase(),traj2.widths(),traj2.x(),traj2.p())

    # Off-diagonal element between trajectories on the same adiabatic
    # state
    if  traj1.state() == traj2.state():
        # Adiabatic energy
        return traj1.energy(traj1.state()) * nuc_ovrlp

    # off-diagonal matrix elements between trajectories on different
    # elecronic states
    elif traj1.state() != traj2.state():
        # Derivative coupling
        fij = traj1.derivative(traj2.state())
        v = np.dot(fij, 2.*traj1.kecoef*
                          dirac.deldx(nuc_ovrlp,traj1.x(),
                                       traj2.widths(),traj2.x(),traj2.p()))
        return v * nuc_ovrlp
    else:
        print('ERROR in v_integral -- argument disagreement')
        return 0j


def ke_integral(traj1, traj2, nuc_ovrlp=None):
    """ Returns < delta(R-R1) | T | g2 > """
    if traj1.state() != traj2.state():
        return 0j

    else:
        if nuc_ovrlp is None:
            nuc_ovrlp = dirac.overlap(traj1.x(),
                                 traj2.phase(),traj2.widths(),traj2.x(),traj2.p())

        ke = dirac.deld2x(nuc_ovrlp,traj1.x(),
                                 traj2.widths(),traj2.x(),traj2.p())

        return -sum(ke * traj1.kecoef)


def sdot_integral(traj1, traj2, nuc_ovrlp=None):
    """ Returns < delta(R-R1) | d/dt g2 >

    Note that for now we have to pass nuc_ovrlp=1 to deldx and deldp so
    that these functions do not multiply the result by the overlap
    of traj1 and traj2. This isn't ideal, but will do for now.
    """
    if traj1.state() != traj2.state():
        return 0j
    else:
        if nuc_ovrlp is None:
            nuc_ovrlp = dirac.overlap(traj1.x(),
                                 traj2.phase(),traj2.widths(),traj2.x(),traj2.p())

        sdot = (np.dot(traj2.velocity(),
                       dirac.deldx(nuc_ovrlp,traj1.x(),
                          traj2.widths(),traj2.x(),traj2.p())) +
                np.dot(traj2.force(),
                       dirac.deldp(nuc_ovrlp,traj1.x(),traj2.x())) +
                1j * traj2.phase_dot() * nuc_ovrlp)

        return sdot
