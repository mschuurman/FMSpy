"""
Routines for the continuous spawning algorithm.

Schematic:

  start, ti
    |
   \/        spawn_forward
parent(s,ti) -------------> parent(s,ts)
                                 |
             spawn_backward     \/
child(s',ti) <------------- child(s',ts)

1. The spawn routine is called with parent_i at time t0.
2. If parent_i is coupled to another.
"""
import numpy as np
import nomad.core.glbl as glbl
import nomad.core.log as log
import nomad.core.step as step
import nomad.core.surface as evaluate
import nomad.adapt.utilities as utils

coup_hist = []

def spawn(master, dt):
    """Propagates to the point of maximum coupling, spawns a new
    basis function, then propagates the function to the current time."""
    global coup_hist

    basis_grown   = False
    current_time  = master.time
    # list of added trajectories

    # we want to know the history of the coupling for each trajectory
    # in order to assess spawning criteria -- make sure coup_hist has a slot
    # for every trajectory
    if len(coup_hist) < master.n_traj():
        n_add = master.n_traj() - len(coup_hist)
        for i in range(n_add):
            coup_hist.append(np.zeros((glbl.propagate['n_states'], 3)))

    #--------------- iterate over all trajectories in bundle ---------------------
    for i in range(master.n_traj()):
        # only live trajectories can spawn
        if not master.traj[i].alive:
            continue

        for st in range(glbl.propagate['n_states']):
            # can only spawn to different electronic states
            if master.traj[i].state() == st:
                continue

            # compute magnitude of coupling to state j
            coup = abs(master.traj[i].coupling(master.traj[i].state(), st))
            coup_hist[i][st,:] = np.roll(coup_hist[i][st,:],1)
            coup_hist[i][st,0] = coup

            # if we satisfy spawning conditions, begin spawn process
            if spawn_trajectory(master, i, st, coup_hist[i][st,:],
                                current_time):
                # we're going to messing with this trajectory -- mess with a copy
                parent = master.traj[i].copy()

                # propagate the parent forward in time until coupling maximized
                [success, child, parent_spawn, spawn_time,
                 exit_time] = spawn_forward(parent, st, current_time, dt)

                # set the spawn attempt in master, even if spawn failed (avoid repeated fails)
                master.traj[i].last_spawn[st] = spawn_time
                master.traj[i].exit_time[st]  = exit_time

                if success:
                    # at this point, child is at the spawn point. Propagate
                    # backwards in time until we reach the current time
                    child_spawn = child.copy()
                    # need electronic structure at current geometry -- on correct state
                    evaluate.update_pes_traj(child)
                    spawn_backward(child, spawn_time, current_time, -dt)
                    bundle_overlap = utils.overlap_with_bundle(child, master)
                    if not bundle_overlap:
                        basis_grown = True
                        master.add_trajectory(child)
                        child_spawn.label = master.traj[-1].label # a little hacky...
                        utils.write_spawn_log(current_time, spawn_time, exit_time,
                                                  parent_spawn, child_spawn)
                    else:
                        log.print_message('spawn_bad_step',
                                                 ['overlap with bundle too large'])

    # let caller known if the basis has been changed
    return basis_grown


def spawn_forward(parent, child_state, initial_time, dt):
    """Propagates the parent forward (into the future) until the coupling
    decreases."""
    parent_state    = parent.state()
    current_time    = initial_time
    spawn_time      = initial_time
    exit_time       = initial_time
    child_created   = False
    parent_at_spawn = None
    child_at_spawn  = None

    coup = np.zeros(3)
    log.print_message('spawn_start',
                             [parent.label, parent_state, child_state])

    while True:
        coup                = np.roll(coup,1)
        coup[0]             = abs(parent.coupling(parent_state, child_state))
        child_attempt       = parent.copy()
        child_attempt.set_state(child_state)
        adjust_success      = utils.adjust_child(parent, child_attempt,
                                                 parent.derivative(parent_state,
                                                                   child_state))
        sij = abs(glbl.master_int.nuc_overlap(parent, child_attempt))

        # if the coupling has already peaked, either we exit with a successful
        # spawn from previous step, or we exit with a fail
        if np.all(coup[0] < coup[1:]):
            sp_str = 'no [decreasing coupling]'
            log.print_message('spawn_step',
                                     [current_time, coup[0], sij, sp_str])

            if child_created:
                log.print_message('spawn_success', [spawn_time])
                child_at_spawn.exit_time[parent_state] = current_time
            else:
                log.print_message('spawn_failure', [current_time])

            # exit, we're done trying to spawn
            exit_time = current_time
            break

        # coupling still increasing
        else:
            # try to set up the child
            if not adjust_success:
                sp_str = 'no [momentum adjust fail]'
            elif sij < glbl.spawning['spawn_olap_thresh']:
                sp_str = 'no [overlap too small]'
            elif not np.all(coup[0] > coup[1:]):
                sp_str = 'no [decreasing coupling]'
            else:
                spawn_time                              = current_time
                child_created                           = True
                parent_at_spawn                         = parent.copy()
                child_at_spawn                          = child_attempt.copy()
                child_at_spawn.last_spawn[parent_state] = spawn_time
                child_at_spawn.amplitude                = 0j
                child_at_spawn.parent                   = parent.label
                child_at_spawn.label                    = parent.label
                sp_str                                  = 'yes'

            log.print_message('spawn_step',
                                     [current_time, coup[0], sij, sp_str])

            step.step_trajectory(parent, current_time, dt)
            current_time = current_time + dt

    return child_created, child_at_spawn, parent_at_spawn, spawn_time, exit_time


def spawn_backward(child, current_time, end_time, dt):
    """Propagates the child backwards in time until the current time
    is reached."""
    nstep = int(round( np.abs((current_time-end_time) / dt) ))

    back_time = current_time
    for i in range(nstep):
        step.step_trajectory(child, back_time, dt)
        back_time = back_time + dt
        log.print_message('spawn_back', [back_time])


def spawn_trajectory(bundle, traj_index, spawn_state, coup_h, current_time):
    """Checks if we satisfy all spawning criteria."""

    traj = bundle.traj[traj_index]

    # Return False if:
    # if insufficient population on trajectory to spawn
    if abs(traj.amplitude) < glbl.spawning['spawn_pop_thresh']:
        return False

    # we have already spawned to this state
    if current_time <= traj.exit_time[spawn_state]:
        return False

    # there is insufficient coupling
    if abs(traj.coupling(traj.state(), spawn_state)) < glbl.spawning['spawn_coup_thresh']:
        return False

    # if coupling is decreasing
    if abs(coup_h[0]) < abs(coup_h[1]):
        return False

    # if we already have sufficient overlap with a function on the
    # spawn_state
    if utils.max_nuc_overlap(bundle, traj_index,
                             overlap_state=spawn_state) > glbl.propagate['sij_thresh']:
        return False

    return True


def in_coupled_regime(bundle):
    """Checks if we are in spawning regime."""
    for i in range(bundle.n_traj()):
        for st in range(glbl.propagate['n_states']):
            if st != bundle.traj[i].state():
                if abs(bundle.traj[i].coupling(bundle.traj[i].state(), st)) > glbl.spawning['spawn_coup_thresh']:
                    return True

    return False
