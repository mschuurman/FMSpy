"""
Compute integrals over trajectories traveling on vibronic potentials
"""
import numpy as np
import nomad.compiled.nuclear_gaussian as nuclear
import nomad.compiled.nuclear_vibronic as vibronic

# Let FMS know if overlap matrix elements require PES info
overlap_requires_pes = False

# Let propagator know if we need data at centroids to propagate
require_centroids = False

# Determines the Hamiltonian symmetry
hermitian = True

# functional form of bra function ('dirac_delta', 'gaussian')
basis = 'gaussian'

def v_integral(t1, t2, nuc_ovrlp=None):
    """Returns potential coupling matrix element between two trajectories."""
    # evaluate just the nuclear component (for re-use)
    if nuc_ovrlp is None:
        nuc_ovrlp = nuclear.overlap(t1.phase(),t1.widths(),t1.x(),t1.p(),
                                    t2.phase(),t2.widths(),t2.x(),t2.p())

    states = np.sort(np.array([t1.state(), t2.state()]))
    v_total = complex(0.,0.)

    # roll through terms in the hamiltonian
    for i in range(glbl.interface.ham.nterms):

        if np.array_equal(states,glbl.interface.ham.stalbl[i,:]-1):
            # adiabatic states in diabatic basis -- cross terms between orthogonal
            # diabatic states are zero
            [s1,s2] = glbl.interface.ham.stalbl[i,:]-1
            v_term = complex(1.,0.) * glbl.interface.ham.coe[i]
            for q in range(len(glbl.interface.ham.order[i])):
                qi      =  glbl.interface.ham.mode[i][q]
                v_term *=  vibronic.prim_v_integral(glbl.interface.ham.order[i][q],
                           t1.widths()[qi],t1.x()[qi],t1.p()[qi],
                           t2.widths()[qi],t2.x()[qi],t2.p()[qi])
            v_total += v_term

    return v_total * nuc_ovrlp

