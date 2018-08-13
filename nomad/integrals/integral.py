"""
Routines for computing integrals.
"""
import numpy as np
import nomad.core.timings as timings
import nomad.integrals.centroid as centroid

class Integral:
    """Class constructor for the Bundle object."""
    def __init__(self, ansatz):
        self.type     = ansatz
        self.centroid = []
        self.centroid_required = []
        self.ints = __import__('nomad.integrals.gaussian_global', fromlist=['a'])
        self.ints_eval = __import__('nomad.integrals.'+str(self.type),fromlist=['a'])

        self.hermitian            = self.ints.hermitian
        self.require_centroids    = self.ints_eval.require_centroids
        self.overlap_requires_pes = False 

    @timings.timed
    def elec_overlap(self, bra_traj, ket_traj):
        """Calculates the electronic overlap."""
        return self.ints.elec_overlap(bra_traj, ket_traj)

    @timings.timed
    def nuc_overlap(self, bra_traj, ket_traj):
        """Calculates the nuclear overlap."""
        return self.ints.nuc_overlap(bra_traj, ket_traj)

    @timings.timed
    def traj_overlap(self, bra_traj, ket_traj, nuc_ovrlp=None):
        """Calculates the trajectory overlap."""
        return self.ints.traj_overlap(bra_traj, ket_traj, nuc_ovrlp=nuc_ovrlp)

    @timings.timed
    def s_integral(self, bra_traj, ket_traj, nuc_ovrlp=None):
        """Calculates the overlap integral between two trajectories."""
        return self.ints.s_integral(bra_traj, ket_traj, nuc_ovrlp=nuc_ovrlp)

    @timings.timed
    def t_integral(self, bra_traj, ket_traj, nuc_ovrlp=None):
        """Calculates the kinetic energy integral between two trajectories."""
        return self.ints.t_integral(bra_traj, ket_traj, nuc_ovrlp=nuc_ovrlp)

    @timings.timed
    def v_integral(self, bra_traj, ket_traj, nuc_ovrlp=None):
        """Calculates the potential energy integral between two
        trajectories."""
        if self.require_centroids:
            return self.ints_eval.v_integral(bra_traj, ket_traj,
                                        self.centroid[bra_traj.label][ket_traj.label],
                                        nuc_ovrlp=nuc_ovrlp)
        else:
            return self.ints_eval.v_integral(bra_traj, ket_traj, nuc_ovrlp=nuc_ovrlp)

    @timings.timed
    def sdot_integral(self, bra_traj, ket_traj, nuc_ovrlp=None):
        """Calculates the time derivative of the nuclear overlap."""
        return self.ints.sdot_integral(bra_traj, ket_traj, nuc_ovrlp=nuc_ovrlp)

    def wfn_overlap(self, bra_wfn, ket_wfn):
        """Calculates the overall wavefunction overlap."""
        S = 0.
        for i in range(bra_wfn.nalive):
            for j in range(ket_wfn.nalive):
                ii = bra_wfn.alive[i]
                jj = ket_wfn.alive[j]
                S += (self.traj_overlap(bra_wfn.traj[ii], ket_wfn.traj[jj]) *
                                        bra_wfn.traj[ii].amplitude.conjugate() *
                                        ket_wfn.traj[jj].amplitude)
        return S

    def wfn_project(self, bra_traj, ket_wfn):
        """Returns the overlap of the wfn with a trajectory.

        Assumes the amplitude on the trial trajectory is (1.,0.)
        """
        proj = 0j

        for i in range(ket_wfn.nalive + ket_wfn.ndead):
            proj += self.traj_overlap(bra_traj, ket_wfn.traj[i]) * ket_wfn.traj[i].amplitude

        return proj

    def update(self, wfn):
        """Updates the wavefunction information if required."""
        if self.require_centroids:
            self.update_centroids(wfn)

    def add_centroid(self, new_cent):
        """Places the centroid in a centroid array.

        Increases the array size if necessary in order to accomodate data.
        """
        # minimum dimension required to hold this centroid
        ij           = new_cent.parents
        new_dim_cent = max(ij)
        dim_cent     = len(self.centroid)

        # if current array is too small, expand by necessary number of dimensions
        if new_dim_cent > dim_cent:
            for i in range(dim_cent):
                self.centroid[i].extend([None for j in range(new_dim_cent -
                                                                 dim_cent)])

            for i in range(new_dim_cent - dim_cent):
                self.centroid.append([None for j in range(new_dim_cent)])

        self.centroid[ij[0]][ij[1]] = new_cent
        self.centroid[ij[1]][ij[0]] = new_cent

    #------------------------------------------------------
    #
    #  Private Methods
    #
    @timings.timed
    def update_centroids(self, wfn):
        """Increases the centroid 'matrix' to account for new basis functions.

        Called by add_trajectory. Make sure centroid array has sufficient
        space to hold required centroids. Note that n_traj includes alive
        AND dead trajectories -- therefore it can only increase. So, only
        need to check n_traj > dim_cent condition.
        """
        dim_cent = len(self.centroid)

        # number of centroids already correct
        if wfn.n_traj() == dim_cent:
            return

        # n_traj includes living and dead -- this condition should never be satisfied
        if wfn.n_traj() < dim_cent:
            raise ValueError('n_traj() < dim_cent in wfn. Exiting...')

        # ...else we need to add more centroids
        if wfn.n_traj() > dim_cent:
            for i in range(dim_cent):
                self.centroid[i].extend([None for j in range(wfn.n_traj() -
                                                             dim_cent)])
                self.centroid_required[i].extend([None for j in range(wfn.n_traj() -
                                                                      dim_cent)])

            for i in range(wfn.n_traj() - dim_cent):
                self.centroid.append([None for j in range(wfn.n_traj())])
                self.centroid_required.append([None for j in range(wfn.n_traj())])

        for i in range(wfn.n_traj()):
            for j in range(i):
                # now check to see if needed index has an existing trajectory
                # if not, copy trajectory from one of the parents into the
                # required slots
                if self.centroid[i][j] is None and (wfn.traj[i].alive and wfn.traj[j].alive):
                    self.centroid[i][j] = centroid.Centroid(traj_i=wfn.traj[i],
                                                            traj_j=wfn.traj[j])
                    self.centroid[i][j].update_x(wfn.traj[i],wfn.traj[j])
                    self.centroid[i][j].update_p(wfn.traj[i],wfn.traj[j])
                    self.centroid[j][i] = self.centroid[i][j]
                self.centroid_required[i][j] = self.is_required(wfn.traj[i],wfn.traj[j])
                self.centroid_required[j][i] = self.is_required(wfn.traj[j],wfn.traj[i])

    def is_required(self, traj1, traj2):
        """Documentation to come"""
        return traj1.alive and traj2.alive
