"""
General routines for all spawning algorithms.
"""
import numpy as np
import nomad.math.constants as constants
import nomad.core.glbl as glbl
import nomad.core.log as log


def adjust_child(parent, child, scale_dir):
    """Adjust the child momentum so that child and parent have the same
    energy

    1. First try to scale the momentum along the NAD vector direction
    2. If that fails, scale the momentum uniformly
    """
    e_parent = parent.classical()
    e_child  = child.classical()

    # determine the magnitude of the KE correction
    ke_goal  = e_parent - child.potential()
 
    if ke_goal < 0:
        return False

    # try to scale the momentum along the scale direction
    ke_child = child.kinetic()
    scale_vec  = scale_dir

    scale_norm = np.linalg.norm(scale_vec)
    if scale_norm > constants.fpzero:
        scale_vec = scale_vec / scale_norm
    else:
        # if scale_dir is zero, scale momentum uniformly
        scale_vec = np.ones(len(scale_dir))
        scale_vec = scale_vec / np.linalg.norm(scale_vec)

    p_child = child.p()
    # scale the momentum along the scale_vec direction
    p_para = np.dot(p_child, scale_vec) * scale_vec
    p_perp = p_child - p_para
 
    # the kinetic energy is given by:
    # KE = (P . P) * / (2M)
    #    = (x * p_para + p_perp).(x * p_para + p_perp) / (2M)
    #    = x^2 * (p_para.p_para) / 2M + 2.*x*(p_para.p_perp) / 2M + (p_perp.p_perp) / 2M
    #    = x^2 * KE_para_para + x * KE_para_perp + KE_perp_perp
    #inv_mass     = 1. / (2. * child.masses())
    ke_coef      =     child.kecoef
    ke_para_para =     np.dot( p_para, p_para * ke_coef )
    ke_para_perp = 2.* np.dot( p_para, p_perp * ke_coef )
    ke_perp_perp =     np.dot( p_perp, p_perp * ke_coef )

    # scale p_para by x so that KE == ke_goal
    # (ke_para_para)*x^2 + (ke_para_perp)*x + (ke_perp_perp - ke_goal) = 0
    # solve quadratic equation
    a = ke_para_para
    b = ke_para_perp
    c = ke_perp_perp - ke_goal

    discrim = b**2 - 4.*a*c
    if discrim < 0:
        return False

    if abs(a) > constants.fpzero:
        x = (-b + np.sqrt(discrim)) / (2.*a)
    elif abs(b) > constants.fpzero:
        x = -c / b
    else:
        x = 0.

    p_new = x*p_para + p_perp

    child.update_p(p_new)

    return True


def overlap_with_bundle(traj, bundle):
    """Checks if trajectory has significant overlap with any trajectories
    already in the bundle."""
    t_overlap_bundle = False

    for i in range(bundle.n_traj()):
        if bundle.traj[i].alive:

            if traj.state() != bundle.traj[i].state():
                sij = 0j
            else:
                sij = glbl.master_int.traj_overlap(traj, bundle.traj[i])
            if abs(sij) > glbl.propagate['sij_thresh']:
                t_overlap_bundle = True
                break

    return t_overlap_bundle


def max_nuc_overlap(bundle, overlap_traj, overlap_state=None):
    """Returns the maximum overlap between the nuclear component of
    traj_i, and other trajectories in the bundle.

    If overlap_state is specified, only consider overlap with
    trajectories on state overlap_state.
    """
    max_sij = 0.
    for j in range(bundle.n_traj()):
        if bundle.traj[j].alive and j != overlap_traj:
            if overlap_state is None or bundle.traj[j].state() == overlap_state:
                max_sij = max(max_sij, abs(glbl.master_int.nuc_overlap(
                                                 bundle.traj[overlap_traj],
                                                 bundle.traj[j])))

    return max_sij


def write_spawn_log(entry_time, spawn_time, exit_time, parent, child):
    """Packages data to print to the spawn log."""
    # add a line entry to the spawn log
    data = [entry_time, spawn_time, exit_time]
    data.extend([parent.label, parent.state(), child.label, child.state()])
    data.extend([parent.kinetic(), child.kinetic(), parent.potential(),
                 child.potential()])
    data.extend([parent.classical(), child.classical()])
    log.print_spawn_log(data)
