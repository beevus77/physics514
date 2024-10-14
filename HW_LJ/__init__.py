# import necessary dependencies
import numpy as np
import lennard_jones as lj


if __name__ == "__main__":

    # simulation parameters
    Nx = 5; Ny = 5; N = Nx * Ny  # set particles onto a grid initially
    L = 5
    Nstep = 10000
    rcut = 2.5  # a usual choice for the cutoff radius


    vx = np.zeros(N)
    vy = np.zeros(N)
    rx = np.zeros(N)
    ry = np.zeros(N)

    rxlog = np.zeros([Nstep, N])
    rylog = np.zeros([Nstep, N])
    vxlog = np.zeros([Nstep, N])
    vylog = np.zeros([Nstep, N])

    lj.initialize_positions_and_velocities(rx, ry, vx, vy, Nx, Ny, L)
    for i in range(Nstep):
      dV_drx = np.zeros(N)
      dV_dry = np.zeros(N)
      lj.compute_forces(rx, ry, dV_drx, dV_dry, N, L, rcut)

      # TODO: propagate using velocity Verlet
      lj.euler(rx, ry, vx, vy, dV_drx, dV_dry)
      # lj.velocity_verlet(rx, ry, vx, vy, dV_drx, dV_dry)

      # make sure we're still in the box
      lj.rebox(rx, ry, L)

      # keep track for printing
      rxlog[i] = rx
      rylog[i] = ry
      vxlog[i] = vx
      vylog[i] = vy

      # get some observables
      Epot = lj.compute_potential_energy(rx, ry, rcut, L)
      Ekin = lj.compute_kinetic_energy(vx, vy)
      print(i, Epot, Ekin, Epot + Ekin)
      
    # print result
    lj.print_result(rxlog, rylog, vxlog, vylog)
