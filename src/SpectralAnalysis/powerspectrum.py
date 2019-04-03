import numpy as np
import math
import scipy
import scipy.optimize
import scipy.fftpack

import LightCurve

def add_ps(psall, method='avg'):
    pssum = np.zeros(len(psall[0].ps))
    for x in psall:
        pssum = pssum + x.ps

    if method.lower() in ['average', 'avg', 'mean']:
        pssum = pssum / len(psall)

    psnew = PowerSpectrum()
    psnew.freq = psall[0].freq
    psnew.ps = pssum
    psnew.n = psall[0].n
    psnew.df = psall[0].df
    psnew.norm = psall[0].norm
    return psnew

class PowerSpectrum(LightCurve.Lightcurve):
    def __init__(self, lc=None, counts=None, nphot=None, norm='leahy', m=1, verbose=False):

        self.norm = norm

        if isinstance(lc, LightCurve.Lightcurve) and counts is None:
            pass

        elif not lc is None and not counts is None:
            if verbose == True:
                print("You put in a standard light curve (I hope). Converting to object of type Lightcurve")
            lc = LightCurve.Lightcurve(lc, counts, verbose=verbose)
        else:
            self.freq = None
            self.ps = None
            self.df = None
            return

        if nphot is None:
            nphots = np.sum(lc.counts)
        else:
            nphots = nphot

        # nel = np.round(lc.tseg/lc.res)
        nel = len(lc.counts)

        df = 1.0 / lc.tseg
        fnyquist = 0.5 / (lc.time[1] - lc.time[0])

        # do Fourier transform
        fourier = scipy.fftpack.fft(lc.counts)
        # f2 = fourier.conjugate()
        # ff = f2*fourier
        # fr = np.array([x.real for x in ff])
        fr = np.abs(fourier) ** 2.

        if norm.lower() in ['leahy']:
            # self.ps = 2.0*fr[0: int(nel/2)]/nphots
            p = np.abs(fourier[:nel / 2]) ** 2.
            self.ps = 2. * p / np.sum(lc.counts)

        elif norm.lower() in ['rms']:
            # self.ps = 2.0*lc.tseg*fr/(np.mean(lc.countrate)**2.0)
            p = fr[:nel / 2 + 1] / np.float(nel ** 2.)
            self.ps = p * 2. * lc.tseg / (np.mean(lc.counts) ** 2.0)

        elif norm.lower() in ['variance', 'var']:
            self.ps = self.ps * nphots / len(lc.counts) ** 2.0

        self.df = df
        self.freq = np.arange(len(self.ps)) * self.df + self.df / 2.
        self.nphots = nphots
        self.n = len(lc.counts)
        self.m = m

    def compute_fractional_rms(self, minfreq, maxfreq, bkg=None):
        minind = self.freq.searchsorted(minfreq)
        maxind = self.freq.searchsorted(maxfreq)
        if bkg is None:
            powers = self.ps[minind:maxind]
        else:
            powers = self.ps[minind:maxind] - bkg
        if self.norm == "leahy":
            rms = np.sqrt(np.sum(powers) / (self.nphots))
        elif self.norm == "rms":
            rms = np.sqrt(np.sum(powers * self.df))

        return rms

    def rebinps(self, res, verbose=False):
        # frequency range of power spectrum
        flen = (self.freq[-1] - self.freq[0])
        # calculate number of new bins in rebinned spectrum
        bins = np.floor(flen / res)
        bindf = flen / bins
        binfreq, binps, dt = self._rebin_new(self.freq, self.ps, res, method='mean')
        newps = PowerSpectrum()
        newps.freq = binfreq
        newps.ps = binps
        newps.df = dt
        newps.nphots = binps[0]
        newps.n = 2 * len(binps)
        newps.m = self.m * int(bindf / self.df)
        return newps

    def rebin_log(self, f=0.01):
        df = self.df
        minfreq = self.freq[0] - 0.5 * df
        maxfreq = self.freq[-1]
        binfreq = [minfreq, minfreq + df]
        while binfreq[-1] <= maxfreq:
            binfreq.append(binfreq[-1] + df * (1. + f))
            df = binfreq[-1] - binfreq[-2]

        binps, bin_edges, binno = scipy.stats.binned_statistic(self.freq, self.ps, statistic="mean", bins=binfreq)

        nsamples = np.array([len(binno[np.where(binno == i)[0]]) for i in range(np.max(binno))])
        df = np.diff(binfreq)
        binfreq = binfreq[:-1] + df / 2.
        return binfreq, binps, nsamples

    def findmaxpower(self):
        psfiltered = filter(lambda x: x >= 100.0, self.ps)
        maxpow = max(psfiltered)
        return maxpow

    def checknormal(self, freq, ps):
        # checks the normalization of a power spectrum above fnyquist/10 Hz
        fmin = max(freq) / 10.0
        minind = np.array(freq).searchsorted(fmin)
        psnew = ps[minind:-1]
        normlevel = np.average(psnew)
        normvar = np.var(psnew)

        return normlevel, normvar