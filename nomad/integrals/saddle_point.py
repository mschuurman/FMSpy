"""
Compute saddle-point integrals over trajectories traveling on adiabataic
potentials

This currently uses first-order saddle point.
"""
import numpy as np
import nomad.core.glbl as glbl
import nomad.compiled.nuclear_gaussian as nuclear

# Let propagator know if we need data at centroids to propagate
require_centroids = True

# Determines the Hamiltonian symmetry
hermitian = False

# Returns functional form of bra function ('dirac_delta', 'gaussian')
basis = 'gaussian'


def v_integral(t1, t2, centroid, nuc_ovrlp=None):
    """Returns potential coupling matrix element between two trajectories."""
    # if we are passed a single trajectory, this is a diagonal
    # matrix element -- simply return potential energy of trajectory
    if t1.label == t2.label:
        # Adiabatic energy
        v = t1.energy(t1.state())
        # DBOC
        if glbl.iface_params['coupling_order'] == 3:
            v += t1.scalar_coup(t1.state(), t2.state())
        return v

    if nuc_ovrlp is None:
        nuc_ovrlp = nuc_overlap(t1, t2)

    # off-diagonal matrix element, between trajectories on the same
    # state (this also requires the centroid be present)
    elif t1.state() == t2.state():
        # Adiabatic energy
        v = centroid.energy(t1.state()) * nuc_ovrlp
        # DBOC
        if glbl.iface_params['coupling_order'] == 3:
            v += centroid.scalar_coup(t1.state(), t2.state()) * nuc_ovrlp
        return v

    # [necessarily] off-diagonal matrix element between trajectories
    # on different electronic states
    elif t1.state() != t2.state():
        # Derivative coupling
        fij = centroid.derivative(t1.state(), t2.state())
        v = 2.*np.vdot(fij, t1.kecoef *
                       nuclear.deldx(nuc_ovrlp,t1.widths(),t1.x(),t1.p(),
                                               t2.widths(),t2.x(),t2.p()))
        # Scalar coupling
        if glbl.iface_params['coupling_order'] > 1:
            v += centroid.scalar_coup(t1.state(), t2.state()) * nuc_ovrlp
        return v

    else:
        print('ERROR in v_integral -- argument disagreement')
        return 0j
