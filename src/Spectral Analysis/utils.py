from __future__ import with_statement
from collections import defaultdict
import numpy as np
import scipy

def choice_hack(data, p=None, size = 1):
    weights = p
    # all choices are at equal probability if no weights given
    if weights == None:
        weights = [1.0 / float(len(data)) for x in range(len(data))]
    if weights == None:
        weights = [1.0 / float(len(data)) for x in range(len(data))]

    if not np.sum(weights) == 1.0:
        if np.absolute(weights[0]) > 1.0e7 and sum(weights) == 0:
            weights = [1.0 / float(len(data)) for x in range(len(data))]
        else:
            raise Exception("Weights entered do not add up to 1! This must not happen!")

    # Compute edges of each bin
    edges = []
    etemp = 0.0
    for x, y in zip(data, weights):
        etemp = etemp + y
        edges.append(etemp)

    # if the np.sum of all weights does not add up to 1, raise an Exception
    if size == 1:
        randno = np.random.rand()

        # Else make sure that size is an integer
        # and make a list of random numbers
        try:
            randno = [np.random.rand() for x in np.arange(size)]
        except TypeError:
            raise TypeError("size should be an integer!")

        choice_index = np.array(edges).searchsorted(randno)
        choice_data = np.array(data)[choice_index]

        return choice_data

class TwoPrint(object):
    """
    Print both to a screen and to a file.
    Parameters
    ----------
    filename : string
        The name of the file to save to.
    """

    def __init__(self,filename):
        self.file = open(filename, "w")
        self.filename = filename
        self.file.write("##\n")
        self.close()
        return

    def __call__(self, printstr):
        """
        Print to a the screen and a file at the
        same time.
        Parameters
        ----------
        printstr : string
            The string to be printed to file and screen.
        """
        print(printstr)
        self.file = open(self.filename, "a")
        self.file.write(printstr + "\n")
        self.close()
        return

    def close(self):
        self.file.close()
        return

def autocorr(x, nlags = 100, fourier=False, norm = True):

    """ Computes the autocorrelation function,
    i.e. the correlation of a data set with itself.
    To do this, shift data set by one bin each time and compute correlation for
    the data set with itself, shifted by i bins
    If the data is _not_ correlated, then the autocorrelation function is the delta
    function at lag = 0
    The autocorrelation function can be computed explicitly, or it can be computed
    via the Fourier transform (via the Wiener-Kinchin theorem, I think)
    Parameters
    ----------
    x : {list, array-like}
        The input data to autocorrelate.
    nlags : int, optional, default 100
        The number of lags to compute,
    fourier: boolean, optional, default False
        If True, use the Fourier transform to compute the ACF (True),
        otherwise don't.
    norm : boolean, optional, default True
        If True, normalize the the ACF to 1
    Returns
    -------
    rnew : array-like
        The autocorrelation function of the data in x
    """

    # empty list for the ACF
    r = []
    # length of the data set
    xlen = len(x)

    # shift data set to a mean=0 (otherwise it comes out wrong)
    x1 = np.copy(x) - np.mean(x)
    x1 = list(x1)

    # add xlen zeros to the array of the second time series (to be able to shift it)
    x1.extend(np.zeros(xlen))

    # if not fourier == True, compute explicitly
    if not fourier:
        # loop over all lags
        for a in range(nlags):
            # make a np.array of 2*xlen zeros to store the data set in
            x2 = np.zeros(len(x1))
            # put data set in list, starting at lag a
            x2[a:a+xlen] = x-np.mean(x)
            # compute autocorrelation function for a, append to list r
            r.append(sum(x1*x2)/((xlen - a)*np.var(x)))

    # else compute autocorrelation via Fourier transform
    else:
        # Fourier transform of time series
        fourier = scipy.fft.fft(x-np.mean(x))
        # take conjugate of Fourier transform
        f2 = fourier.conjugate()
        # multiply both together to get the power spectral density
        ff = f2*fourier
        # extract real part
        fr = np.array([b.real for b in ff])
        ps = fr
        # autocorrelation function is the inverse Fourier transform
        # of the power spectral density
        r = scipy.ifft(ps)
        r = r[:nlags+1]
    # if norm == True, normalize everything to 1
    if norm:
        rnew = r/(max(r))
    else:
        rnew = r
    return rnew

def _checkinput(gti):
    if len(gti) == 2:
        try:
            iter(gti[0])
        except TypeError:
            return [gti]
    return gti

class Data(object):
    def __init__(self):
        raise Exception("Don't run this! Use subclass RXTEData or GBMData instead!")

    # Filter out photons that are outside energy thresholds cmin and cmax
    def filterenergy(self, cmin, cmax):
        self.photons= [s for s in self.photons if s._in_range(cmin, cmax)]

    def filtergti(self, gti=None):
        if not gti:
            gti = self.gti
        gti= _checkinput(gti)
        filteredphotons = []
        times = np.array([t.unbarycentered for t in self.photons])
        for g in gti:
            tmin = times.searchsorted(g[0])
            tmax = times.searchsorted(g[1])
            photons = self.photons[tmin:tmax]
            filteredphotons.extend(photons)
        self.photons = filteredphotons

    def filterburst(self, bursttimes, blen=None, bary=False):
        tstart= bursttimes[0]
        tend = bursttimes[1]
        if blen is None:
            blen = tend - tstart

        #tunbary = np.array([s.unbarycentered for s in self.photons])
        time = np.array([s.time for s in self.photons])

        if bary == False:
            tunbary = np.array([s.time for s in self.photons])
            stind = tunbary.searchsorted(tstart)
            eind = tunbary.searchsorted(tend)
        else:
            stind = time.searchsorted(tstart)
            eind = time.searchsorted(tend)

        self.burstphot = self.photons[stind:eind]

    def obsbary(self, poshist):

        # photon times of arrival in MET seconds
        tobs = np.array([s.time for s in self.photons])
        # barycentered satellite position history time stamps in MET seconds
        phtime = np.array([s.phtime for s in poshist.satpos])
        # Interpolate barycentering correction to photon TOAs
        tcorr = np.interp(tobs, phtime, poshist.tdiff)
        # barycentered photon TOAs in TDB seconds since MJDREFI
        ctbarytime = (tobs + tcorr)
        ctbarymet = ctbarytime + self.mjdreff*8.64e4  # barycentered TOA in MET seconds

        # barycenter trigger time the same way as TOAs
        trigcorr = np.interp(self.trigtime, phtime, poshist.tdiff)
        trigcorr = (self.trigtime + trigcorr)

        # barycentered photon TOAs in seconds since trigger time
        ctbarytrig = ctbarytime - trigcorr
        # barycentered photon TOAs as Julian Dates
        ctbaryjd = ctbarytime/8.64e4 + self.mjdrefi + 2400000.5

        # return dictionary with TOAs in different formats
        ctbary = {'barys': ctbarytime, 'trigcorr': trigcorr, 'barymet': ctbarymet, 'barytrig': ctbarytrig, 'baryjd': ctbaryjd}
        return ctbary

class Photon(object):
    def __init__(self, time, energy):
         self.time = time
         self.energy=energy

    def mission2mjd(self, mjdrefi, mjdreff, timezero=0.0):
        self.mjd = (mjdrefi + mjdreff) + (self.time + timezero)/86400.0

    def _in_range(self, lower, upper):
        if lower <= self.energy <= upper:
            return True
        else:
            return False

        