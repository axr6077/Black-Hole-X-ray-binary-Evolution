from __future__ import division
from scipy.integrate import simps
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.stats import kstest
from scipy.stats import norm
import numpy as np
import scipy.fftpack
import scipy.signal
from src.BHKickUtils import tools as gwt
import cmath
import scipy.interpolate
from matplotlib.mlab import griddata

def plot(time,ht1,title='Time domain waveform'):
    '''
    Plots a single waveform
    '''
    fig = plt.figure()
    plt1 = fig.add_subplot(111)
    plt1.plot(list(time), list(ht1))
    axes = plt.gca()
    axes.set_xlim([-0.02,0.02])
    plt.show()


class noise():
    def __init__(self):

        self.aligochar = self.LIGOnoisedata()

    def LIGOnoisedata(self):
        """Returns an interpolated function for the noise data produced by LIGO.
        INPUT: formatted file illustrating aLIGO design minimums.

        Simply reads in the data file from the file specified in open(),
        and interpolates the ASD from the file.

        OUTPUT: Interpolated data for the ASD of the sensitivity of aLIGO.
        """
        with open("ZERO_DET_high_P.txt") as file:
            filearr = file.readlines()
            dataarr = []
            for line in filearr:
                strpoint = line.strip('\n').lstrip(' ').split('   ')
                valpoint = []
                for i in strpoint:
                    valpoint.append(np.float(i))
                dataarr.append(valpoint)

            data = np.array(dataarr)
            return data


class waveform(object):
    def __init__(self, infile):
        self.infile = infile
        self.data = self.waveformdata(self.infile)
        self.fftdata = self.data[0]
        self.f = self.fftdata[0]
        self.hf = self.fftdata[1]
        self.t = self.data[1]
        self.ht = self.data[2]
        self.fft_func = self.data[3]
        self.dT = self.data[4]

    def waveformdata(self, filename):
        indir = '../surrogatemodel/'
        waveformarray = np.loadtxt(indir + filename, dtype=float, usecols=(0, 1, 2))
        N = len(waveformarray[:, 1])
        dT = (waveformarray[1, 0] - waveformarray[0, 0])

        window = scipy.signal.tukey(N, alpha=0.1)
        amp = (waveformarray[:, 1])  # *window

        fftlist = gwt.fft(amp, 1 / dT)
        fft_func = interp1d(fftlist[0], fftlist[1])

        return fftlist, waveformarray[:, 0], amp, fft_func, dT


class velowaveform(object):
    def __init__(self, timedata, ampdata, velshift, sigma, tshift, phase):

        self.velshift = velshift
        self.data = self.veloshiftdata(timedata, ampdata, sigma, tshift, phase)
        self.fftdata = self.data[0]
        self.fft_func = self.data[1]
        self.t = self.data[3]
        self.ht = self.data[2]
        self.f = self.fftdata[0]
        self.hf = self.fftdata[1]
        self.timeshift = tshift
        self.dT = self.data[4]
        self.stime = self.data[5]

    def veloshiftdata(self, timedata, ampdata, sigma, tshift, phase):

        times = timedata
        amp = ampdata
        N = len(times)
        dT = times[1] - times[0]

        shifttime = [times[0]]
        mini = 1
        ind = 0
        for n in range(0, len(times) - 1):
            if abs(times[n]) < abs(mini):
                mini = times[n]
                ind = n
            shifttime.append(shifttime[-1] + (times[n + 1] - times[n]) * (self.velocityshift(times[n], sigma) + 1))

        stimes = []
        for time in shifttime:
            stimes.append(time + tshift - shifttime[ind])  # TODO REMOVE the ind additions

        shiftedwave = interp1d(stimes, amp, bounds_error=False, fill_value=0)

        window = scipy.signal.tukey(N, alpha=0.1)
        amp = shiftedwave(times)  # *window

        fftlist = gwt.fft(amp, 1 / dT)

        phaseamp = fftlist[1] * np.exp(1j * phase)
        fftlist = [fftlist[0], phaseamp]

        fft_func = interp1d(fftlist[0], fftlist[1])

        return fftlist, fft_func, amp, times, dT, shifttime

    def velocityshift(self, time, sigma):
        mrel = mass * 4.93 * 10 ** -6
        velo = self.velshift * norm.cdf(time, 0, (sigma * mrel))
        return velo

mass = 60
sigma = 100
velocity = 0.00

if __name__ == '__main__':
    noi = noise()
    wave = waveform("cbc_q1.00_M60_d410_t140.00_p0.00.dat")
    gwt.fft_plot(wave.f, wave.hf, title='Original Waveform')
    olist = []
    tlist = []
    plist = []
    time = 0
    phase = 0
    velowave = velowaveform(wave.t, wave.ht, velocity, sigma, time, phase)
    veloht = gwt.infft(velowave.hf, 1 / velowave.dT)
    print([np.round(time, 6), np.round(phase, 6)])
    gwt.wave_plot(wave.t, wave.ht, velowave.t, veloht)
    times = np.linspace(wave.t[0], wave.t[-1])
    velofunc = interp1d(velowave.t, veloht)
    fig = plt.figure()
    plt1 = fig.add_subplot(111)
    plt1.plot(velowave.t, veloht - wave.ht)
    axes = plt.gca()
    axes.set_xlim([-0.02, 0.02])
    plt.show()