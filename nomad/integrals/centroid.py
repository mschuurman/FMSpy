"""
The Centroid object and its associated functions.
"""
import copy
import numpy as np
import nomad.core.timings as timings
import nomad.math.constants as constants


def cent_label(itraj_id, jtraj_id):
    """Returns the centroid id for centroid between traj_i, traj_j"""
    idi          = max(itraj_id, jtraj_id)
    idj          = min(itraj_id, jtraj_id)
    return -((idi * (idi - 1) // 2) + idj + 1)


class Centroid:
    """Class constructor for the Centroid object."""
    def __init__(self, traj_i=None, traj_j=None, nstates=0, states=[-1,-1],
                 dim=0, width=None, label=-1):
        if traj_i is None or traj_j is None:
            # total number of states
            self.nstates = int(nstates)
            # parent states
            self.states = states
            # dimensionality of the centroid
            self.dim     = int(dim)
            # widths of gaussians for each dimension
            if width is None:
                self.width = np.zeros(dim)
            else:
                self.width = np.asarray(width)
            # unique identifier for centroid
            self.label   = label
            # current position of the centroid
            self.pos     = np.zeros(self.dim)
            # current momentum of the centroid
            self.mom     = np.zeros(self.dim)
            # labels of parent trajectories
            self.parents = np.zeros(2, dtype=int)
        else:
            idi          = max(traj_i.label, traj_j.label)
            idj          = min(traj_i.label, traj_j.label)
            self.parents = np.array([idj, idi], dtype=int)
            self.nstates = max(traj_i.nstates,traj_j.nstates)
            self.states  = [traj_i.state(), traj_j.state()]
            self.dim     = max(traj_i.dim, traj_j.dim)
            self.label     = -((idi * (idi - 1) // 2) + idj + 1)
            # now update the position in phase space of the centroid
            # if wid_i == wid_j, this is clearly just the simply mean
            # position.
            wid_i = traj_i.widths()
            wid_j = traj_j.widths()
            self.width = 0.5 * (wid_i + wid_j)
            self.pos = (wid_i*traj_i.x() + wid_j*traj_j.x()) / (wid_i+wid_j)
            self.mom = (wid_i*traj_i.p() + wid_j*traj_j.p()) / (wid_i+wid_j)

        # data structure to hold the data from the interface
        self.pes  = None

    @timings.timed
    def copy(self):
        """Copys a Centroid object with new references."""
        new_cent = Centroid(nstates=self.nstates, states=self.states,
                            dim=self.dim, width=self.width,label=self.label)
        new_cent.pos = copy.deepcopy(self.pos)
        new_cent.mom = copy.deepcopy(self.mom)
        if self.pes is not None:
            new_cent.pes = self.pes.copy()
        return new_cent

    #----------------------------------------------------------------------
    #
    # Functions for setting basic pes information from centroid
    #
    #----------------------------------------------------------------------
    def update_x(self, traj_i, traj_j):
        """Updates the position of the centroid."""
        wid_i    = traj_i.widths()
        wid_j    = traj_j.widths()
        self.pos = (wid_i*traj_i.x() + wid_j*traj_j.x()) / (wid_i+wid_j)

    def update_p(self, traj_i, traj_j):
        """Updates the momentum of the centroid."""
        wid_i    = traj_i.widths()
        wid_j    = traj_j.widths()
        self.mom = (wid_i*traj_i.p() + wid_j*traj_j.p()) / (wid_i+wid_j)

    def update_pes_info(self, new_pes):
        """Updates information about the potential energy surface."""
        self.pes = new_pes.copy()

    #-----------------------------------------------------------------------
    #
    # Functions for retrieving basic pes information from centroid
    #
    #-----------------------------------------------------------------------
    def x(self):
        """Returns the position of the centroid as an array."""
        return self.pos

    def p(self):
        """Returns the momentum of the centroid as an array."""
        return self.mom

    def widths(self):
        """Returns a vector containing the widths of the basis functions
        along each degree of freedom."""
        return self.width

    #--------------------------------------------------------------------
    #
    # Functions to update information about the potential energy surface
    #
    #--------------------------------------------------------------------
    def energy(self, state, geom_chk=True):
        """Returns the potential energies.

        Add the energy shift right here. If not current, recompute them.
        """
        if np.linalg.norm(self.pes.get_data('geom') - self.x()) > 10.*constants.fpzero and geom_chk:
            print('WARNING: trajectory.energy() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.label)+
                  '\ntraj.x()='+str(self.x())+"\npes_geom="+str(self.pes.get_data('geom')))
        #return self.pes.get_data('potential')[state] + glbl.propagate['pot_shift']
        return self.pes.get_data('potential')[state]

    def derivative(self, state_i, state_j, geom_chk=True):
        """Returns the derivative with ket state = rstate.

        Bra state assumed to be the current state.
        """
        if geom_chk and np.linalg.norm(self.pes.get_data('geom') - self.x()) > 10.*constants.fpzero:
            print('WARNING: trajectory.derivative() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.label)+
                  '\ntraj.x()='+str(self.x())+"\npes_geom="+str(self.pes.get_data('geom')))
        return self.pes.get_data('derivative')[:, state_i, state_j]

    def hessian(self, state_i, geom_chk=True):
        """Returns the hessian of the potential on state state_i."""
        if geom_chk and np.linalg.norm(self.pes.get_data('geom') - self.x()) > 10.*constants.fpzero:
            print('WARNING: trajectory.hessian() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.label)+
                  '\ntraj.x()='+str(self.x())+"\npes_geom="+str(self.pes.get_data('geom')))
        return self.pes.get_data('hessian')[:, :, state_i]

    def coupling(self, state_i, state_j, geom_chk=True):
        """Returns the coupling between surfaces state_i and state_j."""
        if geom_chk and np.linalg.norm(self.pes.get_data('geom') - self.x()) > 10.*constants.fpzero:
            print('WARNING: trajectory.coupling() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.label)+
                  '\ntraj.x()='+str(self.x())+"\npes_geom="+str(self.pes.get_data('geom')))
        return self.pes.get_data('coupling')[state_i, state_j]

    def scalar_coup(self, state_i, state_j, geom_chk=True):
        """Returns the scalar coupling for Hamiltonian
        block (self.state,c_state)."""
        if 'scalar_coup' not in self.pes.avail_data():
            return 0.
        if geom_chk and np.linalg.norm(self.pes.get_data('geom') - self.x()) > 10.*constants.fpzero:
            print('WARNING: trajectory.scalar_coup() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.label)+
                  '\ntraj.x()='+str(self.x())+"\npes_geom="+str(self.pes.get_data('geom')))
        return self.pes.get_data('scalar_coup')[state_i, state_j]

    def nact(self, state_i, state_j, geom_chk=True):
        """Returns the derivative coupling between adiabatic states
        block (self.state,c_state)."""
        if 'nac' not in self.pes.avail_data():
            return 0.
        if geom_chk and np.linalg.norm(self.pes.get_data('geom') - self.x()) > 10.*constants.fpzero:
            print('WARNING: trajectory.nact() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.label)+
                  '\ntraj.x()='+str(self.x())+"\npes_geom="+str(self.pes.get_data('geom')))
        return self.pes.get_data('nact')[:,state_i, state_j]

    #------------------------------------------------------------------------
    #
    # Computed quantities from the
    #
    #------------------------------------------------------------------------
    def potential(self):
        """Returns classical potential energy of the centroid."""
        return 0.5 * (self.energy(self.states[0]) + self.energy(self.states[1]))

    def kinetic(self):
        """Returns classical kinetic energy of the centroid."""
        return sum( self.p() * self.p() / (2. * self.masses()))

    def classical(self):
        """Returns the classical energy of the centroid."""
        return self.potential() + self.kinetic()

    def velocity(self):
        """Returns the velocity of the centroid."""
        return self.p() / self.masses()

    def force(self):
        """Returns the gradient of the centroid state."""
        if self.states[0] != self.states[1]:
            return np.zeros(self.dim)
        return -self.derivative(self.states[0], self.states[1])
