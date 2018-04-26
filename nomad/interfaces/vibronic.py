"""
Routines for running a vibronic coupling calculation.
"""
import sys
import copy
import numpy as np
import scipy.linalg as sp_linalg
import nomad.fmsio.glbl as glbl
import nomad.fmsio.fileio as fileio

kecoeff = None
ham = None
nsta = glbl.propagate['n_states']
data_cache = dict()


class Surface:
    """Object containing potential energy surface data."""
    def __init__(self, tag, n_states, t_dim):
        # necessary for array allocation
        self.tag      = tag
        self.n_states = n_states
        self.t_dim    = t_dim

        # these are the standard quantities ALL interface_data objects return
        self.data_keys = []
        # the current geometry
        self.geom      = np.zeros(t_dim)
        # the potential energy (adiabatic or diabatic, depening on integral scheme)
        self.potential = np.zeros(n_states)
        # the derivative of the potentials. 
        # For adiabatic surfaces, diagonal elements
        # are the adiabatic gradients, of diagaonal are the derivative couplings
        # For diabatic surfaces, just the derivative of the diabatic Hamiltonian
        self.deriv     = np.zeros((t_dim, n_states, n_states))
        # the second derivaative of the potentials
        # For adiabatic surfaces, returns the hessians and the derivatives of the
        # derivative couplings
        # For diabatic propagation: second derivative of the diabatic Hamiltonian
        self.deriv2    = np.zeros((t_dim, t_dim, n_states, n_states))
        # return the coupling between the states
        # For adiabatic surfaces, this is the derivative coupling
        # For diabatic surfaces, this is the potential coupling divided by Delta E
        self.coupling = np.zeros((t_dim, n_states, n_states))
        # included for adiabatic coupling only
        self.scalar_coup    = np.zeros((n_states, n_states))
        # nonadiabatic coupling
        self.nac            = np.zeros((t_dim, n_states, n_states))

        # the following will potentially be cleaned up (removed) at a later date:
        self.dat_mat        = np.zeros((n_states, n_states))
        self.adt_mat        = np.zeros((n_states, n_states))
        self.ddat_mat       = np.zeros((t_dim, n_states, n_states))
        self.diabat_pot     = np.zeros((n_states, n_states))
        self.diabat_deriv   = np.zeros((t_dim, n_states, n_states))
        self.diabat_deriv2  = np.zeros((t_dim, t_dim, n_states, n_states))
        self.adiabat_pot    = np.zeros(n_states)
        self.adiabat_deriv  = np.zeros((t_dim, n_states))
        self.adiabat_deriv2 = np.zeros((t_dim, t_dim, n_states))

    def copy(self):
        """Creates a copy of a Surface object."""
        new_info = Surface(self.tag, self.n_states, self.t_dim)

        new_info.data_keys     = copy.copy(self.data_keys)
        new_info.geom          = copy.deepcopy(self.geom)
        new_info.potential     = copy.deepcopy(self.potential)
        new_info.deriv         = copy.deepcopy(self.deriv)
        new_info.deriv2        = copy.deepcopy(self.deriv2)
        new_info.coupling      = copy.deepcopy(self.coupling)
        new_info.scalar_coup    = copy.deepcopy(self.scalar_coup)
        new_info.nac           = copy.deepcopy(self.nac)

        new_info.dat_mat        = copy.deepcopy(self.dat_mat)
        new_info.adt_mat        = copy.deepcopy(self.dat_mat)
        new_info.ddat_mat       = copy.deepcopy(self.ddat_mat)
        new_info.diabat_pot     = copy.deepcopy(self.diabat_pot)
        new_info.diabat_deriv   = copy.deepcopy(self.diabat_deriv)
        new_info.diabat_deriv2  = copy.deepcopy(self.diabat_deriv2)
        new_info.adiabat_pot    = copy.deepcopy(self.adiabat_pot)
        new_info.adiabat_deriv  = copy.deepcopy(self.adiabat_deriv)
        new_info.adiabat_deriv2 = copy.deepcopy(self.adiabat_deriv2)
        return new_info


class VibHam:
    """Object containing the vibronic Hamiltonian parameters."""
    def __init__(self):
        # Parameters
        self.npar   = 0
        self.apar   = None
        self.par    = None
        self.parmap = dict()

        # Hamiltonian terms
        self.nterms       = 0
        self.coe          = None
        self.stalbl       = None
        self.order        = None
        self.mode         = None
        self.freq         = None
        self.mlbl_total   = []
        self.mlbl_active  = []
        self.nmode_total  = 0
        self.nmode_active = 0
        self.freqmap      = dict()
        self.mrange       = None

    def rdgeomfile(self, fname):
        """Reads the labels of the geometry.dat file for ordering purposes."""
        with open(fname, 'r') as infile:
            self.nmode_total = int(infile.readline().split()[0])
            infile.readline()
            for i in range(self.nmode_total):
                lbl = infile.readline().split()[0]
                self.mlbl_total.append(lbl.lower())

    def rdfreqfile(self, fname):
        """Reads and interprets a freq.dat file."""
        with open(fname, 'r') as infile:
            keywords = get_kwds(infile)

        self.nmode_active = len(keywords)
        self.mlbl_active = ['' for i in range(self.nmode_active)]
        self.freq = np.zeros(self.nmode_active)
        for i, kwd in enumerate(keywords):
            self.mlbl_active[i] = kwd[0]
            self.freq[i] = conv(kwd[1:])
            self.freqmap[kwd[0]] = self.freq[i]

    def rdoperfile(self, fname):
        """Reads and interprets an operator file."""
        with open(fname, 'r') as infile:
            keywords = get_kwds(infile)

        parstart = keywords.index(['parameter-section'])
        parend   = keywords.index(['end-parameter-section'])
        self.npar = parend - parstart - 1
        self.apar = ['' for i in range(self.npar)]
        self.par  = np.zeros(self.npar)
        for i in range(self.npar):
            kwd = keywords[parstart + i + 1]
            self.apar[i] = kwd[0]
            if kwd[1] != '=':
                raise NameError('No argument has been given with the keyword: ' +
                                kwd[0])
            self.par[i] = conv(kwd[2:])
            self.parmap[kwd[0]] = self.par[i]

        hamstart = keywords.index(['hamiltonian-section'])
        hamend   = keywords.index(['end-hamiltonian-section'])
        i = hamstart + 1
        opind = []
        while i < hamend:
            kwd = keywords[i]
            if kwd[0] != 'modes' and i == hamstart + 1:
                raise ValueError('The Hamiltonian section must start with the '
                                 'mode specification')
            elif kwd[0] != 'modes':
                break
            while '|' in kwd:
                kwd.remove('|')
            opind += [self.mlbl_total.index(i) for i in kwd[1:]]
            i += 1

        nterms = hamend - i
        coe    = np.zeros(nterms)
        mode   = [[] for i in range(nterms)]
        order  = [[] for i in range(nterms)]
        stalbl = np.zeros((nterms, 2), dtype=int)
        active = np.zeros(nterms, dtype=bool)
        for i in range(nterms):
            kwd = keywords[hamend - nterms + i]
            if '^' in kwd:
                powind = [j for j in range(len(kwd)) if kwd[j] == '^']
                for j in powind:
                    modei = opind[int(kwd[j-1]) - 1]
                    if self.mlbl_total[modei] in self.mlbl_active:
                        mode[i].append(modei)
                        order[i].append(int(kwd[j+1]))
                        active[i] = True
            else:
                powind = [len(kwd)]
                active[i] = True
            coe[i] = get_coe(kwd[:powind[0]-1])
            if '&' not in kwd[-1]:
                raise ValueError('The Hamiltonian term must end with state '
                                 'specification')
            states = sorted(kwd[-1][1:].split('&'))
            stalbl[i] = [int(s) for s in states]

        self.nterms = sum(active)
        self.coe    = coe[active]
        self.stalbl = stalbl[active]
        self.mode   = np.array(mode)[active]
        self.order  = np.array(order)[active]
        self.mrange = [self.mlbl_total.index(i) for i in self.mlbl_active]


def init_interface():
    """Reads geometry.dat, freq.dat and the operator file.

    Note that the order of modes is determined by geometry.dat and
    the active modes are determined from the freq.dat file.
    As such, we must read the labels in geometry.dat followed by
    freq.dat file BEFORE reading the operator file.
    """
    global kecoeff, ham

    # Read in geometry labels, frequency and operator files
    ham = VibHam()

    # if 'geometry.dat' present, read info from there, else, from inputfile
    if glbl.nuclear_basis['geomfile'] != '':
        ham.rdgeomfile(fileio.home_path + '/geometry.dat')
    else:
        ham.nmode_total = len(glbl.nuclear_basis['geometries'][0])
        ham.mlbl_total  = glbl.nuclear_basis['labels']

    # I propose discontinuing 'freq.dat' file. This can be entered in
    # input file. Need way to differentiate between active/inactive modes
    # I presume
    #ham.rdfreqfile(fileio.home_path + '/freq.dat')
    ham.nmode_active = len(glbl.nuclear_basis['freqs'])
    ham.mlbl_active  = ham.mlbl_total
    ham.freq         = np.array(glbl.nuclear_basis['freqs'])
    for i in range(len(ham.freq)):
        ham.freqmap[ham.mlbl_active[i]] = ham.freq[i]

    # operator file will always be a separate file
    ham.rdoperfile(fileio.home_path + '/' + glbl.interface['opfile'])

    # KE operator coefficients, mass- and frequency-scaled normal mode
    # coordinates, a_i = 0.5*omega_i
    kecoeff = np.zeros(ham.nmode_total)
    kecoeff[ham.mrange] = 0.5
    print('kecoeff='+str(kecoeff))

    # Ouput some information about the Hamiltonian
    fileio.print_fms_logfile('string', ['*'*72])
    fileio.print_fms_logfile('string',
                             ['* Vibronic Coupling Hamiltonian Information'])
    fileio.print_fms_logfile('string', ['*'*72])
    fileio.print_fms_logfile('string',
                             ['Operator file: ' + glbl.interface['opfile']])
    fileio.print_fms_logfile('string',
                             ['Number of Hamiltonian terms: ' + str(ham.nterms)])
    string = 'Total no. modes: ' + str(ham.nmode_total)
    fileio.print_fms_logfile('string', [string])

    string = 'No. active modes: ' + str(ham.nmode_active)
    fileio.print_fms_logfile('string', [string])

    fileio.print_fms_logfile('string', ['Active mode labels:'])
    for i in range(ham.nmode_active):
        string = str(i+1) + ' ' + ham.mlbl_active[i]
        fileio.print_fms_logfile('string', [string])

    fileio.print_fms_logfile('string', ['Active mode frequencies (a.u.):'])
    for i in range(ham.nmode_active):
        string = str(i+1) + ' ' + str(ham.freq[i])
        fileio.print_fms_logfile('string', [string])


def evaluate_trajectory(traj, t=None):
    """Evaluates the trajectory."""
    global data_cache

    label = traj.label
    geom  = traj.x()

    # Calculation of the diabatic potential matrix
    diabpot = calc_diabpot(geom)

    # Calculation of the nuclear derivatives of the diabatic potential
    diabderiv1 = calc_diabderiv1(geom)

    # Calculatoin of the hessian matrix for each sub-block of the diabatic potential 
    # matrix
    diabderiv2 = calc_diabderiv2(geom)

    # Calculation of the Laplacian of the diabatic potential wrt the
    # nuclear DOFs
    diablap = calc_diablap(geom)

    #** load the data into the pes object to pass to trajectory **

    t_data           = Surface(label, nsta, ham.nmode_active)
    t_data.geom      = geom

    t_data.diabat_pot     = diabpot
    t_data.diabat_deriv   = diabderiv1
    t_data.diabat_deriv2  = diabderiv2

    t_data.data_keys     = ['geom','poten','deriv','deriv2','coupling','nac',
                            'diabat_pot','diabat_deriv','diabat_deriv2']

    if glbl.variables['surface_rep'] == 'adiabatic':
        # Calculation of the adiabatic potential vector and ADT matrix
        adiabpot, datmat = calc_dat(label, diabpot)

        # Calculation of the NACT matrix
        nactmat = calc_nacts(adiabpot, datmat, diabderiv1)

        # Calculation of the gradients (for diagonal elements) and derivative 
        # couplings (off-diagonal elements)
        adiabderiv1 = calc_adiabderiv1(datmat, diabderiv1)

        # Calculation of the hessians (for diagonal elements) and derivative of
        # derivative couplings (off-diagonal elements)
        adiabderiv2 = calc_adiabderiv2(datmat, diabderiv2)

        # Calculation of the scalar couplings terms (SCTs)
        # Note that in order to calculate the SCTs, we get the gradients of the
        # diagonal Born-Oppenheimer corrections (DBOCs). Consequently, we
        # save these as a matter of course.
        sctmat, dbocderiv1 = calc_scts(adiabpot, datmat, diabderiv1,
                                   nactmat, adiabderiv1, diablap)

        t_data.potential   = adiabpot
        t_data.deriv       = [np.diag(adiabderiv1[m]) for m in
                             range(ham.nmode_total)] + nactmat
        t_data.deriv2      = adiabderiv2
        t_data.coupling    = nactmat
        t_data.nac         = nactmat
        t_data.scalar_coup = 0.5*sctmat #account for the 1/2 prefactor in the EOMs

        # adiabatic -> diabatic and diabatic -> adiabatic transformation matrices
        t_data.dat_mat        = datmat
        t_data.adt_mat        = datmat.T
        t_data.adiabat_pot    = adiabpot
        t_data.adiabat_deriv  = adiabderiv1
        t_data.adiabat_deriv2 = adiabderiv2

        t_data.data_keys.extend(['adiabat_pot','adiabat_deriv'])

    else:
        # Calculation of the adiabatic potential vector and ADT matrix
        adiabpot, datmat = calc_dat(label, diabpot)
        # Calculation of the NACT matrix
        nactmat = calc_nacts(adiabpot, datmat, diabderiv1)
        # determine the effective diabatic coupling: Hij / (Ei - Ej)
        diab_effcoup     = calc_diabeffcoup(diabpot)

        t_data.potential = np.array([diabpot[i,i] for i in range(nsta)])
        t_data.deriv     = diabderiv1 
        t_data.deriv2    = diabderiv2
        t_data.coupling  = diab_effcoup
        t_data.nac       = nactmat

    data_cache[label] = t_data
    return t_data

def evaluate_centroid(traj, t=None):
    """Evaluates the centroid.

    At the moment, this function is just evaluate_trajectory.
    """
    return evaluate_trajectory(traj, t)


#----------------------------------------------------------------------
#
# Private functions (called only within the module)
#
#----------------------------------------------------------------------
def conv(val_list):
    """Takes a list and converts it into atomic units.

    The input list can either be a single float or a float followed by a
    comma and the units to convert from. Currently supported units are
    Hartrees (au), electron volts (ev) and wavenumbers (cm).
    """
    if len(val_list) == 1:
        return float(val_list[0])
    elif len(val_list) > 2 and val_list[1] == ',':
        if val_list[2] == 'au':
            return float(val_list[0])
        elif val_list[2] == 'ev':
            return float(val_list[0]) / glbl.constants['au2ev']
        elif val_list[2] == 'cm':
            return float(val_list[0]) / glbl.constants['au2cm']
        else:
            raise ValueError('Unknown units:', val_list[2])
    else:
        raise ValueError('Unknown parameter format:', val_list)


def get_kwds(infile):
    """Reads a file and returns keywords.

    By default, everything is converted to lowercase.
    """
    delim = ['=', ',', '(', ')', '[', ']', '{', '}', '|', '*', '/', '^']
    rawtxt = infile.readlines()
    kwds = [[] for i in range(len(rawtxt))]

    for i, line in enumerate(rawtxt):
        if '#' in line:
            line = line[:line.find('#')]
        for d in delim:
            line = line.replace(d, ' ' + d + ' ')
        kwds[i] = line.lower().split()

    while [] in kwds:
        kwds.remove([])
    return kwds


def get_coe(val_list):
    """Gets a coefficient from a list of factors"""
    if len(val_list) % 2 != 1:
        raise ValueError('Coefficient specification must have odd number of '
                         'terms including arithmetic operators.')
    val_list.insert(0, '*')
    coeff = 1
    for j in range(len(val_list) // 2):
        try:
            fac = float(val_list[2*j+1])
        except ValueError:
            fac = ham.parmap[val_list[2*j+1]]
        if val_list[2*j] == '*':
            coeff *= fac
        elif val_list[2*j] == '/':
            coeff /= fac
        else:
            raise ValueError('Aritmetic operator must be \'*\' or \'/\'')
    return coeff


def calc_diabpot(q):
    """Constructs the diabatic potential matrix for a given nuclear
    geometry q.

    N.B. the terms held in the coe, ord and stalbl contribute only
    to the lower-triangle of the diabatic potential matrix.
    """
    diabpot = np.zeros((nsta, nsta))

    # Fill in the lower-triangle
    for i in range(ham.nterms):
        s1, s2 = ham.stalbl[i] - 1
        if ham.mode[i] == []:
            diabpot[s1,s2] += ham.coe[i]
        else:
            diabpot[s1,s2] += ham.coe[i] * np.prod(q[ham.mode[i]]**ham.order[i])

    # Fill in the upper-triangle
    diabpot += diabpot.T - np.diag(diabpot.diagonal())
    return diabpot


def calc_dat(label, diabpot):
    """Diagonalises the diabatic potential matrix to yield the adiabatic
    potentials and the adiabatic-to-diabatic transformation matrix."""
    adiabpot, datmat = sp_linalg.eigh(diabpot)

    if label in data_cache:
        # Ensure phase continuity from geometry to another
        datmat *= np.sign(np.dot(datmat.T, data_cache[label].dat_mat).diagonal())
    else:
        # Set phase convention that the greatest abs element in dat column
        # vector is positive
        datmat *= np.sign(datmat[range(len(adiabpot)),
                                 np.argmax(np.abs(datmat), axis=0)])
    return adiabpot, datmat

def calc_ddat(label, q, diabpot, dat_mat):
    """Returns the derviative of the diabatic to adiabatic transformation 
       matrix via numerical differentiation"""
    ddat_mat = np.zeros((ham.nmode_total,nsta,nsta))
    dx = 0.001

    return ddat_mat

def calc_diabderiv1(q):
    """Calculates the 1st derivatives of the elements of the diabatic
    potential matrix wrt the nuclear DOFs."""
    diabderiv1 = np.zeros((ham.nmode_total, nsta, nsta))

    # Fill in the lower-triangle
    for i in range(ham.nterms):
        nmodes = len(ham.mode[i])
        s1, s2 = ham.stalbl[i] - 1
        if nmodes == 1:
            m = ham.mode[i][0]
            o = ham.order[i][0]
            diabderiv1[m,s1,s2] += ham.coe[i] * o * q[m]**(o - 1)
        elif nmodes > 1:
            m = ham.mode[i]
            o = np.array(ham.order[i])
            qcol = np.repeat(q[m], nmodes).reshape(nmodes, nmodes)
            fac = np.prod((qcol - np.diag(q[m]) + np.diag(o)) *
                          (q[m]**(o - 1))[:,np.newaxis], axis=0)
            diabderiv1[m,s1,s2] += ham.coe[i] * fac

    # Fill in the upper-triangle
    diabderiv1[ham.mrange] += (np.transpose(diabderiv1[ham.mrange], axes=(0,2,1)) -
                               [np.diag(diabderiv1[m].diagonal()) for m in
                                ham.mrange])
    return diabderiv1

def calc_diabderiv2(q):
    """Calculates the 2nd derivatives of the elements of the diabatic
    potential matrix wrt the nuclear DOFs."""
    diabderiv2 = np.zeros((ham.nmode_total, ham.nmode_total, nsta, nsta))

    # Fill in the lower-triangle
    for i in range(ham.nterms):
        nmodes = len(ham.mode[i])
        s1, s2 = ham.stalbl[i] - 1
        m = ham.mode[i]
        o = np.array(ham.order[i])

        # first do diagonal d^2/dx^2 terms
        elem = [ind for ind,val in enumerate(o) if val > 1]
        nelem = len(elem)
        if nelem > 0:
            prim = np.repeat(q[m]**o, nelem).reshape(len(m), nelem)
            derv = o[elem]*(o[elem]-1)*(q[elem]**(o[elem]-2))
            for j in range(nelem):
                prim[j,elem[j]] = derv[j]
                trm = ham.coe[i] * np.prod(prim[j])
                diabderiv2[m[elem[j]],m[elem[j]],s1,s2] += trm
                if s1 != s2:
                    diabderiv2[m[elem[j]],m[elem[j]],s2,s1] += trm

        #now do bilinear terms
        elem = [ind for ind,val in enumerate(o) if val > 0]
        nelem = len(elem)
#        print("s1,s2="+str(s1)+' '+str(s2))
#        print("o="+str(o))
#        print("m="+str(m))
#        print("len(m)="+str(len(m)))
#        print("q[m]="+str(q[m]))
#        print("q[m]**o="+str(q[m]**o))
#        print("nelem="+str(nelem))
#        print("nterm="+str(nterm))
#        if nelem > 1:
#            nterm = int(nelem * (nelem-1) / 2)
#            print("nterm="+str(nterm))
#            prim = np.repeat(q[m]**o, nterm).reshape(len(m), nterm)
#            derv = o[elem]*q[elem]**(o[elem]-1)
#            icnt = 0
#            for j in range(nelem):
#                for k in range(j):
#                    prim[icnt,elem[j]] = derv[j]
#                    prim[icnt,elem[k]] = derv[k]
#                    trm = ham.coe[i] * np.prod(prim[icnt])
#                    diabderiv2[m[elem[j]],m[elem[k]],s1,s2] += trm
#                    if elem[j] != elem[k]:
#                        diabderiv2[m[elem[k]],m[elem[j]],s1,s2] += trm
#                    if s1 != s2:
#                        diabderiv2[m[elem[j]],m[elem[k]],s2,s1] += trm
#                        if elem[j] != elem[k]:
#                            diabderiv2[m[elem[k]],m[elem[j]],s2,s1] += trm
#                    icnt+=1

    return diabderiv2

def calc_diabeffcoup(diabpot):
    """Calculates the effective diabatic coupling between diabatic states i, j via
       eff_coup = Hij / (H[i,i] - H[j,j])
    """
    eff_coup = np.zeros((1, nsta, nsta))

    demat = np.array([[max([glbl.constants['fpzero'],diabpot[i,i]-diabpot[j,j]],key=abs) 
                           for i in range(nsta)] for j in range(nsta)])
    eff_coup[0] = np.divide(diabpot - np.diag(diabpot.diagonal()), demat)

    return eff_coup


def calc_nacts(adiabpot, datmat, diabderiv1):
    """Calculates the matrix of non-adiabatic coupling terms from the
    adiabatic potentials, the ADT matrix and the nuclear derivatives of
    the diabatic potentials.

    Equation used: F_aij = [S (d/dX_a W) S^T]_ij / (V_j - V_i)
    F_aij = < psi_i | d/dX_a psi_j >: non-adiabatic coupling terms
    W: diabatic potential matrix
    S^T: matrix of eigenvectors of W
    V: vector of adiabatic energies
    """
    # Fill in the matrix
    nactmat = np.zeros((ham.nmode_total, nsta, nsta))
    fac = -np.subtract.outer(adiabpot, adiabpot) + np.eye(nsta)
    for m in ham.mrange:
        nactmat[m] = np.dot(np.dot(datmat.T, diabderiv1[m]), datmat) / fac

    # Subtract the diagonal to make sure it is zero
    nactmat -= [np.diag(nactmat[m].diagonal()) for m in range(ham.nmode_total)]
    return nactmat

def calc_adiabderiv1(datmat, diabderiv1):
    """Calculates the gradients of the adiabatic potentials.

    Equation used: d/dX V_ii = (S{d/dX W}S^T)_ii
    """
    # Get the diagonal elements of the matrix
    adiabderiv1 = np.zeros((ham.nmode_total, nsta))
    for m in ham.mrange:
        adiabderiv1[m] = np.dot(np.dot(datmat.T, diabderiv1[m]),
                                datmat).diagonal()
    return adiabderiv1

def calc_adiabderiv2(datmat, diabderiv2):
    """Calculates the hessians of the adiabatic potentials.

    Equation used: d^2/dXidXj V_ii = (S{d^2/dXidXj W}S^T)_ii 
    """
    # Get the diagonal elements of the matrix
    adiabderiv2 = np.zeros((ham.nmode_total, ham.nmode_total, nsta))

    for m in ham.mrange:
        for n in range(m+1):
            adiabderiv2[m,n] = np.dot(np.dot(datmat.T, diabderiv2[m,n]),
                                             datmat).diagonal()
            if m != n:
                adiabderiv2[n,m] = adiabderiv2[m,n]

    return adiabderiv2


def calc_diablap(q):
    """Calculates the Laplacian of the diabatic potential matrix wrt
    the nuclear DOFs at the point q."""
    diablap = np.zeros((nsta, nsta))

    # Fill in the lower-triangle
    for i in range(ham.nterms):
        nmodes = len(ham.mode[i])
        s1, s2 = ham.stalbl[i] - 1
        if nmodes == 1:
            m = ham.mode[i][0]
            o = ham.order[i][0]
            if o > 1:
                diablap[s1,s2] += ham.coe[i] * o * (o - 1) * q[m]**(o - 2)
        elif nmodes > 1:
            m = ham.mode[i]
            o = np.array(ham.order[i])
            if np.amax(o) > 1:
                q2col = np.repeat(q[m]**2, nmodes).reshape(nmodes, nmodes)
                fac = np.prod((q2col - np.diag(q[m]**2) + np.diag(o * (o - 1))) *
                          (q[m]**(o - 2))[:,np.newaxis], axis=0)
                diablap[s1,s2] += ham.coe[i] * np.sum(fac)

    # Fill in the upper-triangle
    diablap += diablap.T - np.diag(diablap.diagonal())
    return diablap

def calc_scts(adiabpot, datmat, diabderiv1, nactmat, adiabderiv1, diablap):
    """Calculates the scalar coupling terms.

    Uses the equation:
    G = (d/d_X F) - F.F

    d/d_X F_ij = d/dX (xi_ij*T_ij), where, xi_ij = (V_j-V_i)^-1
                                           T = S {d/dX W} S^T

    Additionally, the gradients of the on-diagonal SCTs (DBOCs)
    are calculated

    In the following:

    ximat      <-> xi
    tmat       <-> T
    deltmat    <-> d/dX T
    delximat   <-> d/dX xi
    delnactmat <-> d/dX F
    fdotf      <-> F.F
    """
    #-------------------------------------------------------------------
    # (1) Construct d/dX F (delnactmat)
    #-------------------------------------------------------------------

    # (a) deltmat = S {Del^2 W} S^T + -FS{d/dX W}S^T + S{d/dX W}S^TF
    # tmp1 <-> S {Del^2 W} S^T
    tmp1 = np.dot(np.dot(datmat.T, diablap), datmat)

    # tmp2 <-> -F S{d/dX W}S^T
    mat2 = [np.dot(np.dot(np.dot(nactmat[m], datmat.T), diabderiv1[m]), datmat)
            for m in ham.mrange]
    tmp2 = -np.sum(ham.freq[:,np.newaxis,np.newaxis] * mat2, axis=0)

    # tmp3 <-> S{d/dX W}S^T F
    mat3 = [np.dot(np.dot(np.dot(datmat.T, diabderiv1[m]), datmat), nactmat[m])
            for m in ham.mrange]
    tmp3 = np.sum(ham.freq[:,np.newaxis,np.newaxis] * mat3, axis=0)

    # deltmat
    deltmat = tmp1 + tmp2 + tmp3

    # (b) delximat
    ## This is slightly slower for nsta == 2
    #delximat = np.zeros((ham.nmode_total, nsta, nsta))
    #for m in ham.mrange:
    #    delximat[m] = -(np.subtract.outer(adiabderiv1[m], adiabderiv1[m]) /
    #                    (np.subtract.outer(adiabpot, adiabpot) ** 2 +
    #                     np.eye(nsta)))
    delximat = np.zeros((ham.nmode_total, nsta, nsta))
    for m in ham.mrange:
        for i in range(nsta):
            for j in range(nsta):
                if i != j:
                    delximat[m,i,j] = ((adiabderiv1[m,i] - adiabderiv1[m,j]) /
                                       (adiabpot[j] - adiabpot[i])**2)

    # (c) tmat
    tmat = np.zeros((ham.nmode_total, nsta, nsta))
    for m in ham.mrange:
        tmat[m] = np.dot(np.dot(datmat.T, diabderiv1[m]), datmat)

    # (d) ximat
    ## This is slightly slower for nsta == 2
    #ximat = 1. / (-np.subtract.outer(adiabpot, adiabpot) +
    #              np.eye(nsta)) - np.eye(nsta)
    ximat = np.zeros((nsta, nsta))
    for i in range(nsta):
        for j in range(nsta):
            if i != j:
                ximat[i,j] = 1. / (adiabpot[j] - adiabpot[i])

    # (f) delnactmat_ij = delximat_ij*tmat_ij + ximat_ij*deltmat_ij (i.ne.j)
    delnactmat = np.sum(delximat * tmat, axis=0) + ximat * deltmat

    #-------------------------------------------------------------------
    # (2) Construct F.F (fdotf)
    #-------------------------------------------------------------------
    matf = [np.dot(nactmat[m], nactmat[m]) for m in ham.mrange]
    fdotf = np.sum(ham.freq[:,np.newaxis,np.newaxis] * matf, axis=0)

    #-------------------------------------------------------------------
    # (3) Calculate the scalar coupling terms G = (d/dX F) - F.F
    #-------------------------------------------------------------------
    sctmat = delnactmat - fdotf

    #-------------------------------------------------------------------
    # (4) Calculation of the 1st derivatives of the DBOCs
    #-------------------------------------------------------------------
    dbocderiv1 = np.zeros((ham.nmode_total, nsta))
    for m in ham.mrange:
        dbocderiv1[m] = -2.*np.sum(delnactmat * nactmat[m], axis=0)

    return sctmat, dbocderiv1
