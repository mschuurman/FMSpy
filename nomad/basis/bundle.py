"""
The Bundle object and its associated functions.
"""
import sys
import copy
import numpy as np
import scipy.linalg as sp_linalg
from nomad.dynamics import timings
from nomad.fmsio import glbl as glbl
from nomad.fmsio import fileio as fileio
from nomad.basis import trajectory as trajectory
from nomad.basis import centroid as centroid
from nomad.basis import hamiltonian as fms_ham


class Bundle:
    """Class constructor for the Bundle object."""
    def __init__(self, nstates):
        self.time      = 0.
        self.nalive    = 0
        self.nactive   = 0
        self.ndead     = 0
        self.nstates   = int(nstates)
        self.traj      = []
        self.cent      = []
        self.alive     = []
        self.active    = []
        self.T         = np.zeros((0, 0), dtype=complex)
        self.V         = np.zeros((0, 0), dtype=complex)
        self.S         = np.zeros((0, 0), dtype=complex)
        self.Snuc      = np.zeros((0, 0), dtype=complex)
        self.Sdot      = np.zeros((0, 0), dtype=complex)
        self.Heff      = np.zeros((0, 0), dtype=complex)
        self.traj_ovrlp= np.zeros((0, 0), dtype=complex)
#        try:
#            self.integrals=__import__('nomad.integrals.'+
#                                       self.integral_type,fromlist=['a'])
#        except ImportError:
#            print('BUNDLE INIT FAIL: nomad.integrals.'+self.integral_type)

    @timings.timed
    def copy(self):
        """Copys a Bundle object with new references.

        This method is the simplest way i can see to make a copy
        of a bundle with new references. Overriding deepcopy for
        significantly more work.
        """
        new_bundle = Bundle(self.nstates)
        new_bundle.time   = copy.copy(self.time)
        new_bundle.nalive = copy.copy(self.nalive)
        new_bundle.ndead  = copy.copy(self.ndead)
        new_bundle.alive  = copy.deepcopy(self.alive)
        new_bundle.T      = copy.deepcopy(self.T)
        new_bundle.V      = copy.deepcopy(self.V)
        new_bundle.S      = copy.deepcopy(self.S)
        new_bundle.Snuc   = copy.deepcopy(self.Snuc)
        new_bundle.Sdot   = copy.deepcopy(self.Sdot)
        new_bundle.Heff   = copy.deepcopy(self.Heff)
        new_bundle.traj_ovrlp = copy.deepcopy(self.traj_ovrlp)

        # copy the trajectory array
        for i in range(self.n_traj()):
            traj_i = self.traj[i].copy()
            new_bundle.traj.append(traj_i)

        # copy the centroid matrix
#        if new_bundle.integrals.require_centroids:
        if glbl.integrals.require_centroids:
            for i in range(len(self.cent)):
                new_bundle.cent.append([None for j in range(self.n_traj())])
                for j in range(i):
                    new_bundle.cent[i][j] = self.cent[i][j].copy()
                    new_bundle.cent[j][i] = self.cent[j][i].copy()

        return new_bundle

    def n_traj(self):
        """Returns total number of trajectories."""
        return self.nalive + self.ndead

    @timings.timed
    def add_trajectory(self, new_traj):
        """Adds a trajectory to the bundle."""
        self.nalive         += 1
        self.nactive        += 1
        new_traj.alive       = True
        new_traj.active      = True
        new_traj.label       = self.n_traj() - 1
        self.alive.append(new_traj.label)
        self.active.append(new_traj.label)
        self.traj.append(new_traj)
        self.T       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.V       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.S       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Snuc    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Sdot    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Heff    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.traj_ovrlp = np.zeros((self.nalive, self.nalive), dtype=complex)
#        if self.integrals.require_centroids:
        if glbl.integrals.require_centroids:
            self.create_centroids()

    def add_trajectories(self, traj_list):
        """Adds a set of trajectories to the bundle."""
        for new_traj in traj_list:
            self.nalive         += 1
            self.nactive        += 1
            new_traj.alive       = True
            new_traj.active      = True
            new_traj.label       = self.n_traj() - 1
            self.alive.append(new_traj.label)
            self.active.append(new_traj.label)
            self.traj.append(new_traj)

        self.T       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.V       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.S       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Snuc    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Sdot    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Heff    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.traj_ovrlp = np.zeros((self.nalive, self.nalive), dtype=complex)
#        if self.integrals.require_centroids:
        if glbl.integrals.require_centroids:
            self.create_centroids()

    @timings.timed
    def kill_trajectory(self, label):
        """Moves a live trajectory to the list of dead trajecotries.

        The trajectory will no longer contribute to H, S, etc.
        """
        # Remove the trajectory from the list of living trajectories
        self.alive.remove(label)
        self.traj[label].alive = False
        self.nalive          = self.nalive - 1
        self.ndead           = self.ndead + 1
        # Remove the trajectory from the list of active trajectories
        # iff matching pursuit is not being used
        if not glbl.propagate['matching_pursuit']:
            self.active.remove(tid)
            self.traj[tid].active = False
            self.nactive = self.nactive - 1
        # Reset arrays
        self.T       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.V       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.S       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Snuc    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Sdot    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Heff    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.traj_ovrlp = np.zeros((self.nalive, self.nalive), dtype=complex)

    @timings.timed
    def revive_trajectory(self, label):
        """
        Moves a dead trajectory to the list of live trajectories.
        """
        self.traj[label].alive = True

        # Add the trajectory to the list of living trajectories
        self.alive.insert(label,label)

        self.nalive          = self.nalive + 1
        self.ndead           = self.ndead - 1

        self.T       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.V       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.S       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Snuc    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Sdot    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Heff    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.traj_ovrlp = np.zeros((self.nalive, self.nalive), dtype=complex)

    @timings.timed
    def update_amplitudes(self, dt, update_ham=True, H=None, Ct=None):
        """Updates the amplitudes of the trajectory in the bundle.
        Solves d/dt C = -i H C via the computation of
        exp(-i H(t) dt) C(t)."""
        if update_ham:
            self.update_matrices()

        # if no Hamiltonian is passed, use the current effective
        # Hamiltonian
        if H is None:
            Hmat = self.Heff
        else:
            Hmat = H

        # if no vector of amplitudes are supplied (to propagate),
        # propagate the current amplitudes
        if Ct is None:
            old_amp = self.amplitudes()
        else:
            old_amp = Ct

        new_amp = np.zeros(self.nalive, dtype=complex)

        B = -1j * Hmat * dt

        if self.nalive < 150:
            # Eigen-decomposition
            umat = sp_linalg.expm2(B)
        else:
            # Pade approximation
            umat = sp_linalg.expm(B)

        new_amp = np.dot(umat, old_amp)

        for i in range(len(self.alive)):
            self.traj[self.alive[i]].update_amplitude(new_amp[i])

    @timings.timed
    def centroid_required(self, traj_i, traj_j):
        """Determine if centroid is required for integral. Data at centroid will
           NOT be computed if:
           a) the nuclear overlap between trajectories is below threshold
        """
#        if not self.integrals.require_centroids:
        if not glbl.integrals.require_centroids:
            return False
        else:
            return True

    @timings.timed
    def renormalize(self):
        """Renormalizes the amplitudes of the trajectories in the bundle."""
        norm_factor = 1. / np.sqrt(self.norm())
        for i in range(self.n_traj()):
            self.traj[i].update_amplitude(self.traj[i].amplitude * norm_factor)

    def prune(self):
        """Kills trajectories that are dead."""
        for i in range(self.nalive):
            continue
        return False

    def amplitudes(self):
        """Returns amplitudes of the trajectories."""
        return np.array([self.traj[self.alive[i]].amplitude
                         for i in range(len(self.alive))], dtype=complex)

    def set_amplitudes(self, amps):
        """Sets the value of the amplitudes."""
        for i in range(self.nalive):
            self.traj[self.alive[i]].amplitude = amps[i]

    def mulliken_pop(self, label):
        """Returns the Mulliken-like population."""
        mulliken = 0.

        if not self.traj[label].alive:
            return mulliken
        i = self.alive.index(label)
        for j in range(len(self.alive)):
            jj = self.alive[j]
            mulliken += abs(self.traj_ovrlp[i,j] *
                            self.traj[label].amplitude.conjugate() *
                            self.traj[jj].amplitude)
        return mulliken

    @timings.timed
    def norm(self):
        """Returns the norm of the wavefunction """
        return np.dot(np.dot(np.conj(self.amplitudes()),
                      self.S),self.amplitudes()).real

    @timings.timed
    def pop(self):
        """Returns the populations on each of the states."""
        pop    = np.zeros(self.nstates, dtype=complex)
        nalive = len(self.alive)

        # live contribution
        for i in range(nalive):
            ii = self.alive[i]
            state = self.traj[ii].state
            for j in range(nalive):
                jj = self.alive[j]
                if self.traj[jj].state != state:
                    continue
                popij = (self.traj_ovrlp[i,j]  *
                         self.traj[jj].amplitude *
                         self.traj[ii].amplitude.conjugate())
                pop[state] += popij

        pop /= sum(pop)

        return pop.real

    @timings.timed
    def pot_classical(self):
        """Returns the classical potential energy of the bundle.

        Currently only includes energy from alive trajectories
        """
        nalive  = len(self.alive)
        pot_vec = np.array([self.traj[self.alive[i]].potential()
                            for i in range(nalive)])
        return sum(pot_vec)/nalive

    @timings.timed
    def pot_quantum(self):
        """Returns the QM (coupled) potential energy of the bundle.
        Currently includes <live|live> (not <dead|dead>,etc,) contributions...
        """
        return np.dot(np.dot(np.conj(self.amplitudes()),
                             self.V), self.amplitudes()).real
        #Sinv = sp_linalg.pinv(self.S)
        #return np.dot(np.dot(np.conj(self.amplitudes()),
        #                     np.dot(Sinv,self.V)),self.amplitudes()).real

    @timings.timed
    def kin_classical(self):
        """Returns the classical kinetic energy of the bundle."""
        nalive   = len(self.alive)
        kecoef   = glbl.pes.kecoeff
        print("p="+str(self.traj[self.alive[0]].p()))
        print("kecoef="+str(kecoef))
        ke_vec   = np.array([np.dot(self.traj[self.alive[i]].p()**2, kecoef)
                           for i in range(nalive)])
        print("t="+str(sum(ke_vec)))
        return sum(ke_vec)/nalive

    @timings.timed
    def kin_quantum(self):
        """Returns the QM (coupled) kinetic energy of the bundle."""
        return np.dot(np.dot(np.conj(self.amplitudes()),
                             self.T), self.amplitudes()).real
        #Sinv = sp_linalg.pinv(self.S)
        #return np.dot(np.dot(np.conj(self.amplitudes()),
        #                     np.dot(Sinv,self.T)),self.amplitudes()).real

    def tot_classical(self):
        """Returns the total classical energy of the bundle."""
        return self.pot_classical() + self.kin_classical()

    def tot_quantum(self):
        """Returns the total QM (coupled) energy of the bundle."""
        return self.pot_quantum() + self.kin_quantum()

    def overlap(self, other):
        """Returns the overlap integral of the bundle with another
        bundle."""
        S = 0.
        for i in range(self.nalive):
            for j in range(other.nalive):
                ii = self.alive[i]
                jj = other.alive[j]
                S += (glbl.integrals.traj_overlap(self.traj[ii], other.traj[jj]) *
                      self.traj[ii].amplitude.conjugate() *
                      other.traj[jj].amplitude)
#                S += (self.integrals.traj_overlap(self.traj[ii], other.traj[jj]) *
#                      self.traj[ii].amplitude.conjugate() *
#                      other.traj[jj].amplitude)
        return S

    def overlap_traj(self, traj):
        """Returns the overlap of the bundle with a trajectory (assumes the
        amplitude on the trial trajectory is (1.,0.)"""
        ovrlp = 0j
        for i in range(self.nalive+self.ndead):
            ovrlp += (glbl.integrals.traj_overlap(traj,self.traj[i]) *
                      self.traj[i].amplitude)
#            ovrlp += (self.integrals.traj_overlap(traj,self.traj[i]) *
#                      self.traj[i].amplitude)
        return ovrlp

    #----------------------------------------------------------------------
    #
    # Private methods/functions (called only within the class)
    #
    #----------------------------------------------------------------------
    @timings.timed
    def create_centroids(self):
        """Increases the centroid 'matrix' to account for new basis functions.

        Called by add_trajectory. Make sure centroid array has sufficient
        space to hold required centroids. Note that n_traj includes alive
        AND dead trajectories -- therefore it can only increase. So, only
        need to check n_traj > dim_cent condition.
        """
        dim_cent = len(self.cent)

        # number of centroids already correct
        if self.n_traj() == dim_cent:
            return

        # n_traj includes living and dead -- this condition should never be satisfied
       # if self.n_traj() < dim_cent:
            raise ValueError('n_traj() < dim_cent in bundle. Exiting...')

        # ...else we need to add more centroids
        if self.n_traj() > dim_cent:
            for i in range(dim_cent):
                self.cent[i].extend([None for j in range(self.n_traj() -
                                                         dim_cent)])

            for i in range(self.n_traj() - dim_cent):
                self.cent.append([None for j in range(self.n_traj())])

        for i in range(self.n_traj()):
            if self.traj[i].alive:
                for j in range(i):
                    if self.traj[j].alive:
                        # now check to see if needed index has an existing trajectory
                        # if not, copy trajectory from one of the parents into the
                        # required slots
                        if self.cent[i][j] is None:
                            self.cent[i][j] = centroid.Centroid(traj_i=self.traj[i],
                                                                traj_j=self.traj[j])
                            self.cent[j][i] = self.cent[i][j]

    @timings.timed
    def update_centroids(self):
        """Updates the positions of the centroids."""

        for i in range(self.n_traj()):
            if self.traj[i].alive:
                for j in range(i):
                    if self.traj[j].alive:
                        self.cent[i][j].update_x(self.traj[i],self.traj[j])
                        self.cent[j][i].update_x(self.traj[j],self.traj[i])
                        self.cent[i][j].update_p(self.traj[i],self.traj[j])
                        self.cent[j][i].update_p(self.traj[j],self.traj[i])

    @timings.timed
    def update_matrices(self):
        """Updates T, V, S, Sdot and Heff matrices."""
        # make sure the centroids are up-to-date in order to evaluate
        # self.H -- if we need them

#        if self.integrals.require_centroids:
#            (self.traj_ovrlp, self.T, self.V, self.S, self.Snuc, self.Sdot,
#             self.Heff) = fms_ham.hamiltonian(self.integrals, self.traj, self.alive,
#                                              cent_list=self.cent)
        if glbl.integrals.require_centroids:
            (self.traj_ovrlp, self.T, self.V, self.S, self.Snuc, self.Sdot,
             self.Heff) = fms_ham.hamiltonian(self.traj, self.alive,
                                              cent_list=self.cent)
        else:
#            (self.traj_ovrlp, self.T, self.V, self.S, self.Snuc, self.Sdot,
#             self.Heff) = fms_ham.hamiltonian(self.integrals, self.traj, self.alive)
            (self.traj_ovrlp, self.T, self.V, self.S, self.Snuc, self.Sdot,
             self.Heff) = fms_ham.hamiltonian(self.traj, self.alive)


    #------------------------------------------------------------------------
    #
    # Functions to read/write bundle to checkpoint files
    #
    #------------------------------------------------------------------------
    @timings.timed
    def update_logs(self):
        """Updates the log files."""

        np.set_printoptions(precision=8, linewidth=80, suppress=False)

        filename='/tmp/schuurm/nomad/elec_overlap.dat'
        smat = np.array([[glbl.integrals.elec_overlap(self.traj[j],self.traj[i]) for i in range(self.nalive)] for j in range(self.nalive)])
        with open(filename, 'a') as outfile:
            outfile.write('{:9.2f}\n'.format(self.time))
            outfile.write(np.array2string(smat,
                      formatter={'complex_kind':lambda x: '{: 15.8e}'.format(x)})+'\n')

        filename='/tmp/schuurm/nomad/sig.dat'
        smat = np.array([[glbl.integrals.sdot_integral(self.traj[j],self.traj[i],e_only=True) for i in range(self.nalive)] for j in range(self.nalive)])
        with open(filename, 'a') as outfile:
            outfile.write('{:9.2f}\n'.format(self.time))
            outfile.write(np.array2string(smat,
                      formatter={'complex_kind':lambda x: '{: 15.8e}'.format(x)})+'\n')

        filename='/tmp/schuurm/nomad/tau.dat'
        smat = np.array([[glbl.integrals.sdot_integral(self.traj[j],self.traj[i],nuc_only=True) for i in range(self.nalive)] for j in range(self.nalive)])
        with open(filename, 'a') as outfile:
            outfile.write('{:9.2f}\n'.format(self.time))
            outfile.write(np.array2string(smat,
                      formatter={'complex_kind':lambda x: '{: 15.8e}'.format(x)})+'\n')

        filename='/tmp/schuurm/nomad/theta.dat'
        theta = np.array([glbl.integrals.theta(self.traj[i]) for i in range(self.nalive)])
#        with open(filename, 'a') as outfile:
#            outfile.write('{:9.4f}{:15.8f}{:15.8f}{:15.8f}{:15.8f}\n'.format(self.time,theta[0],theta[1],theta[2],theta[3]))
#            outfile.write('{:9.4f}{:15.8f}{:15.8f}{:15.8f}{:15.8f}\n'.format(self.time,theta[0]))

        filename='/tmp/schuurm/nomad/theta_cache.dat'
        theta = np.array([glbl.integrals.theta_cache[self.traj[i].label] for i in range(self.nalive)])
#        with open(filename, 'a') as outfile:
#            outfile.write('{:9.4f}{:15.8f}{:15.8f}{:15.8f}{:15.8f}\n'.format(self.time,theta[0],theta[1],theta[2],theta[3]))
#            outfile.write('{:9.4f}{:15.8f}{:15.8f}{:15.8f}{:15.8f}\n'.format(self.time,theta[0]))

        filename='/tmp/schuurm/nomad/Hd.dat'
        hdmat = self.traj[0].pes_data.diabat_pot
        with open(filename, 'a') as outfile:
            outfile.write('{:9.2f}\n'.format(self.time))
            outfile.write(np.array2string(hdmat,
                      formatter={'real_kind':lambda x: '{: 15.8e}'.format(x)})+'\n')
        

        for i in range(self.n_traj()):
            if not self.traj[i].active:
                continue

            # trajectory files
            if glbl.printing['print_traj']:
                data = [self.time]
                data.extend(self.traj[i].x().tolist())
                data.extend(self.traj[i].p().tolist())
                data.extend([self.traj[i].phase(), self.traj[i].amplitude.real,
                             self.traj[i].amplitude.imag,
                             abs(self.traj[i].amplitude),
                             self.traj[i].state])

                fileio.print_traj_row(self.traj[i].label, 0, data)

                # potential energy
                data = [self.time]
                data.extend([self.traj[i].energy(j)
                             for j in range(self.nstates)])
                fileio.print_traj_row(self.traj[i].label, 1, data)

                # gradients
                data = [self.time]
                data.extend(self.traj[i].derivative(self.traj[i].state,
                                                    self.traj[i].state).tolist())
                fileio.print_traj_row(self.traj[i].label, 7, data)

                # coupling
                data = [self.time]
                data.extend([self.traj[i].coupling_norm(j)
                             for j in range(self.nstates)])
                data.extend([self.traj[i].coup_dot_vel(j)
                             for j in range(self.nstates)])
                fileio.print_traj_row(self.traj[i].label, 2, data)

                # temporary writing of integral debugging
                filename='/tmp/schuurm/nomad/theta.'+str(self.traj[i].label)
                with open(filename, 'a') as outfile:
                    outfile.write('\n'+str(self.time)+'  '+str(glbl.integrals.theta(self.traj[i])))
                filename='/tmp/schuurm/nomad/hessian.'+str(self.traj[i].label)
                hess=self.traj[i].pes_data.diabat_deriv2[:,:,self.traj[i].state, self.traj[i].state]
                dderiv=self.traj[i].pes_data.diabat_deriv2[0,1]
                with open(filename, 'a') as outfile:
                    outfile.write('{:9.2f}\n'.format(self.time))
                    outfile.write(np.array2string(hess,
                      formatter={'complex_kind':lambda x: '{: 15.8e}'.format(x)})+'\n')
                    outfile.write(np.array2string(dderiv,
                      formatter={'complex_kind':lambda x: '{: 15.8e}'.format(x)})+'\n')

            # print pes information relevant to the chosen interface
            if glbl.printing['print_es']:

                # print the interface-specific data
                for key in self.traj[i].pes_data.data_keys:

                    # skip the stuff that is NOT interface speicifc
                    if key in ['geom','poten','deriv']:
                        continue

                    # permanent dipoles
                    if key == 'dipole':
                        data = [self.time]
                        for j in range(self.nstates):
                            data.extend(self.traj[i].pes_data.dipoles[:,j,j].tolist())
                        fileio.print_traj_row(self.traj[i].label, 3, data)

                    # transition dipoles
                    if key == 'tr_dipole':
                        data = [self.time]
                        for j in range(self.nstates):
                            for k in range(j):
                                data.extend(self.traj[i].pes_data.dipoles[:,k,j].tolist())
                        fileio.print_traj_row(self.traj[i].label, 4, data)

                    # second moments
                    if key == 'sec_mom':
                        ncart = 3 
                        data = [self.time]
                        for j in range(self.nstates):
                            diag_mom = [self.traj[i].pes_data.sec_moms[k,k,j] 
                                        for k in range(ncart)]
                            data.extend(diag_mom)
                        fileio.print_traj_row(self.traj[i].label, 5, data)

                    # atomic populations
                    if key == 'atom_pop':
                        data = [self.time]
                        for j in range(self.nstates):
                            data.extend(self.traj[i].pes_data.atom_pop[:,j].tolist())
                        fileio.print_traj_row(self.traj[i].label, 6, data)

        # now dump bundle information ####################################

        # state populations
        data = [self.time]
        data.extend(self.pop().tolist())
        data.append(self.norm())
        fileio.print_bund_row(0, data)

        # bundle energy
        data = [self.time,
                self.pot_quantum(), self.kin_quantum(), self.tot_quantum(),
                self.pot_classical(), self.kin_classical(), self.tot_classical()]
        fileio.print_bund_row(1, data)

        # bundle matrices
        if glbl.printing['print_matrices']:
#            if self.integrals.basis != 'gaussian':
            if glbl.integrals.basis != 'gaussian':
                fileio.print_bund_mat(self.time, 't_ovrlp.dat', self.traj_ovrlp)
            fileio.print_bund_mat(self.time, 'nuc_ovrlp.dat', self.Snuc)
            fileio.print_bund_mat(self.time, 's.dat', self.S)
            fileio.print_bund_mat(self.time, 'h.dat', self.T+self.V)
            fileio.print_bund_mat(self.time, 't.dat', self.T)
            fileio.print_bund_mat(self.time, 'v.dat', self.V)
            fileio.print_bund_mat(self.time, 'heff.dat', self.Heff)
            fileio.print_bund_mat(self.time, 'sdot.dat', self.Sdot)

        # dump full bundle to an checkpoint file
        if glbl.printing['print_chkpt']:
            self.write_bundle(fileio.scr_path + '/last_step.dat','w')

        # wavepacket autocorrelation function
        if glbl.propagate['auto'] and glbl.variables['bundle0'] is not None:
            auto = self.overlap(glbl.variables['bundle0'])
            data = [self.time, auto.real, auto.imag, abs(auto)]
            fileio.print_bund_row(8, data)

    @timings.timed
    def write_bundle(self, filename, mode):
        """Dumps the bundle to file 'filename'.

        Mode is either 'a'(append) or 'x'(new).
        """
        if mode not in ('w','a'):
            raise ValueError('Invalid write mode in bundle.write_bundle')
        ndim    = self.traj[0].dim
        with open(filename, mode) as chkpt:
            # first write out the bundle-level information
            chkpt.write('------------- BEGIN BUNDLE SUMMARY --------------\n')
            chkpt.write('{:10.2f}            current time\n'.format(self.time))
            chkpt.write('{:10d}            live trajectories\n'.format(self.nalive))
            chkpt.write('{:10d}            dead trajectories\n'.format(self.ndead))
            chkpt.write('{:10d}            number of states\n'.format(self.nstates))
            chkpt.write('{:10d}            number of coordinates\n'.format(ndim))

            # information common to all trajectories
            chkpt.write('--------- common trajectory information --------\n')
            chkpt.write('coordinate widths --\n')
            self.traj[0].widths().tofile(chkpt, ' ', '%.4f')
            chkpt.write('\ncoordinate masses --\n')
            self.traj[0].masses().tofile(chkpt, ' ', '%.4f')

            # first write out the live trajectories. The function
            # write_trajectory can only write to a pre-existing file stream
            for i in range(len(self.traj)):
                chkpt.write('\n-------- trajectory {:4d} --------\n'.format(i))
                self.traj[i].write_trajectory(chkpt)
        chkpt.close()

    def read_bundle(self, filename, t_restart):
        """Reads a bundle at time 't_restart' from a chkpt file."""
        try:
            chkpt = open(filename, 'r', encoding='utf-8')
        except:
            raise FileNotFoundError('Could not open: ' + filename)

        # if we're reading from a checkpoint file -- fast-forward
        # to the requested time
        if t_restart != -1:
            t_found = False
            last_pos = chkpt.tell()
            for line in chkpt:
                if 'current time' in line:
                    if float(line[0]) == t_restart:
                        t_found = True
                        break
        # else, we're reading from a last_step file -- and we want to skip
        # over the initial comment line
        else:
            t_found = True
            chkpt.readline()

        if not t_found:
            raise NameError('Could not find time=' + str(t_restart) +
                            ' in ' + str(chkpt.name))

        # read common bundle information
        self.time    = float(chkpt.readline().split()[0])
        nalive  = int(chkpt.readline().split()[0])
        ndead   = int(chkpt.readline().split()[0])
        self.nstates = int(chkpt.readline().split()[0])
        ndim         = int(chkpt.readline().split()[0])

        # the read common info that will be the same for all trajectories
        chkpt.readline()
        # widths
        chkpt.readline()
        widths = np.fromstring(chkpt.readline(), sep=' ', dtype=float)
        # masses
        chkpt.readline()
        masses = np.fromstring(chkpt.readline(), sep=' ', dtype=float)

        # read-in trajectories
        for i in range(nalive + ndead):
            chkpt.readline()
            t_read = trajectory.Trajectory(self.nstates,
                                           ndim,
                                           width=widths,
                                           mass=masses,
                                           label=i,
                                           parent=0)
            t_read.read_trajectory(chkpt)
            self.add_trajectory(t_read)

        # create the bundle matrices
        self.T       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.V       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.S       = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Sdot    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.Heff    = np.zeros((self.nalive, self.nalive), dtype=complex)
        self.traj_ovrlp = np.zeros((self.nalive, self.nalive), dtype=complex)
        # once bundle is read, close the stream
        chkpt.close()
