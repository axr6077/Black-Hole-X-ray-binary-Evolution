import sys, argparse, multiprocessing
import numpy as np
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
import astropy.units as u
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def parser():
    parser = argparse.ArgumentParser(description="Script to estimate Natal kick for a system based on location,\
     proper motion, radial velocity and distance with their uncertainties. This code reports system's potential\
     natal kick based on MC simulations. The probability distribution of all input parameters are assumed to be\
     gaussian at the moment. This code also produces a csv file containing all the realizations from the MC\
     for the natal kick and a png file containing a histogram of this posterior distribution.")

    parser.add_argument('name', type=str, help='Output files root')
    parser.add_argument('ra', type=float, help='System RA (in degrees)')
    parser.add_argument('ra_er', type=float, help='Uncertainty in RA (in degrees)')
    parser.add_argument('dec', type=float, help='System Dec (in degrees)')
    parser.add_argument('dec_er', type=float, help='Uncertainty in Dec')
    parser.add_argument('pm_ra', type=float, help='Proper motion in RA (mas/yr)')
    parser.add_argument('pm_ra_er', type=float, help='Uncertainty in proper motion in RA (mas/yr)')
    parser.add_argument('pm_dec', type=float, help='Proper motion in Dec (mas/yr)')
    parser.add_argument('pm_dec_er', type=float, help='Uncertainty in proper motion in Dec (mas/yr)')
    parser.add_argument('d', type=float, help='Distance to the system (kpc)')
    parser.add_argument('d_er', type=float, help='Uncertainty in distance (kpc)')
    parser.add_argument('v_rad', type=float, help='Systemic radial velocity (km/s)')
    parser.add_argument('v_rad_er', type=float, help='Uncertainty in systemic radial velocity (km/s)')
    parser.add_argument('--niter', type=int, help='Number of iterations in the MC simulations (default: 5000)')
    parser.add_argument('--njobs', type=int,
                        help='Number of threads (parallel) for the simulations (default: number of available kernels on the system)')
    parser.add_argument('--age', type=float,
                        help='Total integration time (in Gyr). Note that Galactic potential is not changing over time. Default 10 Gyr.')
    parser.add_argument('--npoints', type=int, help='Number of orbit points to calculate (Default 10000).')
    args = parser.parse_args()
    return args


def pass_v(ra, dec, d, pm_ra, pm_dec, V, time, numpoints):
    """
    Function to estimate system velocity everytime it passes through the galactic plane.
    This velocity is a proxy for system's natal kick. For details, assumptions and caveats
    see Atri et al. 2019.
    -----
    ra,dec,d,pm_ra,pm_dec,V are source orbit parameters for Galpy

    time is the age to integrate (Gyr)

    numpoints is the number of points in the orbit to integrate.

    """
    o = Orbit([ra * u.deg, dec * u.deg, d * u.kpc, pm_ra * u.mas / u.yr, pm_dec * u.mas / u.yr, V * u.km / u.s],
              radec=True)
    lp = MWPotential2014
    ts = np.linspace(0, time, numpoints) * u.Gyr
    o.integrate(ts, lp)

    pass_t_array = ts[np.where(np.sign(o.z(ts)[:-1]) - np.sign(o.z(ts)[1:]) != 0)[0]]
    results = []
    for pass_t in pass_t_array:
        o2 = Orbit(vxvv=[o.R(pass_t) / 8.0, 0., 1., 0., 0., o.phi(pass_t)], ro=8., vo=220.)
        # results.append(np.sqrt((o.U(pass_t)-o2.U(0)+11.1)**2 + (o.V(pass_t)-o2.V(0)+12.24)**2 + (o.W(pass_t)-o2.W(0)+7.25)**2))
        results.append(
            np.sqrt((o.U(pass_t) - o2.U(0)) ** 2 + (o.V(pass_t) - o2.V(0)) ** 2 + (o.W(pass_t) - o2.W(0)) ** 2))

    return results


def pec_v(ra, dec, d, pm_ra, pm_dec, V):
    o = Orbit([ra * u.deg, dec * u.deg, d * u.kpc, pm_ra * u.mas / u.yr, pm_dec * u.mas / u.yr, V * u.km / u.s],
              radec=True)
    o2 = Orbit(vxvv=[o.R(0.) / 8.0, 0., 1., 0., 0., o.phi(0.)], ro=8., vo=220.)
    current_vel = np.sqrt(
        (o.U(0.) - o2.U(0) + 11.1) ** 2 + (o.V(0.) - o2.V(0) + 12.24) ** 2 + (o.W(0.) - o2.W(0) + 7.25) ** 2)
    return current_vel


def MC_pass_v(RA, RA_err,
              DEC, DEC_err,
              pm_ra, pm_ra_err,
              pm_dec, pm_dec_err,
              V_rad, V_rad_err,
              d, d_err,
              n_iter, n_cores, int_time, orbit_pnts):
    """
    iterator for natal kick
    """
    src_ra = np.random.normal(RA, 0, n_iter)
    src_dec = np.random.normal(DEC, 0, n_iter)
    src_pm_ra = np.random.normal(pm_ra, pm_ra_err, n_iter)
    src_pm_dec = np.random.normal(pm_dec, pm_dec_err, n_iter)
    src_d = np.random.normal(d, d_err, n_iter)
    src_V = np.random.normal(V_rad, V_rad_err, n_iter)
    vel_dist = Parallel(n_jobs=n_cores, verbose=5)(
        delayed(pass_v)(src_ra[i], src_dec[i], src_d[i], src_pm_ra[i], src_pm_dec[i], src_V[i], int_time, orbit_pnts)
        for i in range(n_iter))
    total_dist = np.concatenate(vel_dist[:])
    return total_dist


def MC_pec_v(RA, RA_err,
             DEC, DEC_err,
             pm_ra, pm_ra_err,
             pm_dec, pm_dec_err,
             V_rad, V_rad_err,
             d, d_err,
             n_iter, n_cores, int_time, orbit_pnts):
    src_ra = np.random.normal(RA, 0, n_iter)
    src_dec = np.random.normal(DEC, 0, n_iter)
    src_pm_ra = np.random.normal(pm_ra, pm_ra_err, n_iter)
    src_pm_dec = np.random.normal(pm_dec, pm_dec_err, n_iter)
    src_d = np.random.normal(d, d_err, n_iter)
    src_V = np.random.normal(V_rad, V_rad_err, n_iter)
    vel_dist = Parallel(n_jobs=n_cores, verbose=5)(
        delayed(peculiar)(src_ra[i], src_dec[i], src_d[i], src_pm_ra[i], src_pm_dec[i], src_V[i]) for i in
        range(n_iter))
    total_dist = np.concatenate(vel_dist[:])
    return total_dist


def main():
    input_params = parser()

    if input_params.niter is None:
        input_params.niter = 5000

    if input_params.njobs is None:
        input_params.njobs = multiprocessing.cpu_count()

    if input_params.age is None:
        input_params.age = 10

    if input_params.npoints is None:
        input_params.npoints = 10000

    kicks = MC_pass_v(input_params.ra, input_params.ra_er,
                      input_params.dec, input_params.dec_er,
                      input_params.pm_ra, input_params.pm_ra_er,
                      input_params.pm_dec, input_params.pm_dec_er,
                      input_params.v_rad, input_params.v_rad_er,
                      input_params.d, input_params.d_er,
                      input_params.niter, input_params.njobs,
                      input_params.age, input_params.npoints
                      )
    np.savetxt('natalkick_dist_' + input_params.name + '.csv', kicks, delimiter=",")
    plt.figure()
    plt.hist(kicks, color='green', edgecolor='w', density=True, alpha=0.7)
    plt.ylabel('Probability density', fontsize=12)
    plt.xlabel('Potential natal kick (km/s)', fontsize=12)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='major', length=9)
    plt.tick_params(axis='both', which='minor', length=4.5)
    plt.tick_params(axis='both', which='both', direction='in', right=True, top=True)
    plt.savefig('natalkick_plot_' + input_params.name + '.png', bbox_inches='tight')
    print('System natal kick (km/s):', np.percentile(kicks, 50), '(-',
          np.percentile(kicks, 50) - np.percentile(kicks, 15.9), '/+',
          np.percentile(kicks, 84.1) - np.percentile(kicks, 50), ')')
    print('Reported values are Median +/- 1sigma uncertainties.')
    print('The final MC results are saved in natalkick_dist_' + input_params.name + '.csv')
    print('The distribution is plotted in natalkick_plot_' + input_params.name + '.png')

main()