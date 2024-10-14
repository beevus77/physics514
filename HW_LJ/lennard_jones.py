# import necessary dependencies
import numpy as np


def initialize_positions_and_velocities(rx, ry, vx, vy, Nx, Ny, L):
    """
    Initialize particle positions on a grid and velocities using Box-Muller transform.
  
    Args:
    rx, ry (numpy.ndarray): Arrays to store x and y positions
    vx, vy (numpy.ndarray): Arrays to store x and y velocities
    Nx, Ny (int): Number of particles in x and y directions
    L (float): Box size
  
    Returns:
    None
    """
    dx = L / Nx
    dy = L / Ny
    np.random.seed(0)

    for i in range(Nx):
        for j in range(Ny):
            rx[i * Ny + j] = dx * (i + 0.5)
            ry[i * Ny + j] = dy * (j + 0.5)
      
            u = np.random.random()  # This is box muller
            v = np.random.random()
            vx[i * Ny + j] = np.sqrt(-2 * np.log(u)) * np.cos(2. * np.pi * v)
            vy[i * Ny + j] = np.sqrt(-2 * np.log(u)) * np.sin(2. * np.pi * v)

    # subtract net velocity to avoid global drift
    vxav = sum(vx) / vx.size
    vyav = sum(vy) / vx.size
    vx -= vxav
    vy -= vyav


def force(rsq):
    """
    Calculate the force between two particles (derivative of Lennard-Jones potential).
  
    Args:
    rsq (float): Square of the distance between two particles
  
    Returns:
    float: The force between the particles
    """
    rinv = np.sqrt(1. / rsq)
    return 24 * np.power(rinv,7) * (1 - 2*np.power(rinv,6))


def compute_forces(rx, ry, dV_drx, dV_dry, N, L, rcut):
    """
    Compute the forces between particles using the Lennard-Jones potential.

    This function calculates the forces between all pairs of particles
    in the system using the Lennard-Jones potential. It applies the
    minimum image convention to handle periodic boundary conditions.

    Args:
    rx, ry (numpy.ndarray): Arrays of x and y positions of particles
    dV_drx, dV_dry (numpy.ndarray): Arrays to store the computed forces in x and y directions
    N (int): Number of particles in the system
    L (float): Box size for periodic boundary conditions
    rcut (float): Cutoff radius for force calculation

    Returns:
    None: The function modifies dV_drx and dV_dry in-place to store the computed forces
    """
    rcutsq = rcut * rcut
    for i in range(N):
        for j in range(i):
            dx = rx[i] - rx[j]
            dy = ry[i] - ry[j]
            # minimum image convention
            if (dx > L / 2.): dx = dx - L
            if (dx < -L / 2.): dx = dx + L
            if (dy > L / 2.): dy = dy - L
            if (dy < -L / 2.): dy = dy + L
            # compute the distance
            rsq = dx * dx + dy * dy
            # check if we are < the cutoff radius
            if (rsq < rcutsq):
                # here is the call of the force calculation
                dV_dr = force(rsq)
        
                # here the force is being added to the particle. Note the additional dx
                dV_drx[i] += dx * dV_dr
                dV_drx[j] -= dx * dV_dr
                dV_dry[i] += dy * dV_dr
                dV_dry[j] -= dy * dV_dr


def potential(rsq):
    """
    Calculate the Lennard-Jones potential between two particles.
  
    Args:
    rsq (float): Square of the distance between two particles
  
    Returns:
    float: The potential energy between the particles
    """
    rsqinv = 1. / rsq
    r6inv = rsqinv * rsqinv * rsqinv
    return -4 * r6inv * (1 - r6inv)


def compute_kinetic_energy(vx, vy):
    """
    Compute the total kinetic energy of the system.
  
    Args:
    vx, vy (numpy.ndarray): Arrays of x and y velocities
  
    Returns:
    float: Total kinetic energy
    """
    return 0.5 * sum(vx * vx + vy * vy)


def compute_potential_energy(rx, ry, rcut, L):
    """
    Compute the total potential energy of the system.

    Args:
    rx, ry (numpy.ndarray): Arrays of x and y positions
    rcut (float): Cutoff radius for potential calculation
    L (float): Box size

    Returns:
    float: Total potential energy
    """
    rcutsq = rcut * rcut
    rcutv = potential(rcutsq)  # shift the potential to avoid jump at rc
    Epot = 0.

    for i in range(rx.size):
        for j in range(i):
            dx = rx[i] - rx[j]
            dy = ry[i] - ry[j]
            # minimum image convention
            if (dx > L / 2.): dx = dx - L
            if (dx < -L / 2.): dx = dx + L
            if (dy > L / 2.): dy = dy - L
            if (dy < -L / 2.): dy = dy + L
            rsq = dx * dx + dy * dy
            if rsq < rcutsq:
                Epot += potential(rsq) - rcutv
    return Epot


def euler(rx, ry, vx, vy, dV_drx, dV_dry):
    """
    Perform a single step of the Euler integration method.

    This function updates the positions and velocities of particles using
    the Euler method, which is a first-order numerical integration technique.

    Args:
    rx, ry (numpy.ndarray): Arrays of x and y positions of particles
    vx, vy (numpy.ndarray): Arrays of x and y velocities of particles
    dV_drx, dV_dry (numpy.ndarray): Arrays of x and y components of forces on particles

    Returns:
    None: The function modifies rx, ry, vx, and vy in-place
    """
    deltat = 0.001
    # update the positions
    rx += deltat * vx
    ry += deltat * vy
    
    # update the velocities
    vx -= deltat * dV_drx
    vy -= deltat * dV_dry


def velocity_verlet(rx, ry, vx, vy, dV_drx, dV_dry):
    pass  # TODO: implement


def rebox(rx, ry, L):
    """
    Apply periodic boundary conditions to keep particles within the simulation box.

    This function checks the position of each particle and applies periodic
    boundary conditions to ensure all particles remain within the simulation box.

    Args:
    rx, ry (numpy.ndarray): Arrays of x and y positions of particles
    L (float): Box size for periodic boundary conditions

    Returns:
    None: The function modifies rx and ry in-place
    """
    for i in range(rx.size):
        if rx[i] > L:
            rx[i] = rx[i] - L
        if rx[i] < 0:
            rx[i] = rx[i] + L
        if ry[i] > L:
            ry[i] = ry[i] - L
        if ry[i] < 0:
            ry[i] = ry[i] + L


def print_result(rxlog, rylog, vxlog, vylog):
    """
    Write simulation results to output files.

    This function writes the positions and velocities of particles at each
    timestep to separate output files.

    Args:
    rxlog, rylog (numpy.ndarray): 2D arrays containing x and y positions for all particles at all timesteps
    vxlog, vylog (numpy.ndarray): 2D arrays containing x and y velocities for all particles at all timesteps

    Returns:
    None: The function writes data to 'positions.dat' and 'velocities.dat' files
    """
    with open("HW_LJ/data/positions.dat", 'w') as fr, open("HW_LJ/data/velocities.dat", 'w') as fv:
        for j in range(rxlog.shape[1]):
            for i in range(rxlog.shape[0]):
                fr.write(str(rxlog[i, j]) + " " + str(rylog[i, j]) + '\n')
                fv.write(str(vxlog[i, j]) + " " + str(vylog[i, j]) + '\n')
            fr.write('\n')
            fv.write('\n')
