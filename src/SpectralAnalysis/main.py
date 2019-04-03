import matplotlib.pyplot as plt

## this is just to make plots prettier
## comment out if you don't have seaborn
import seaborn as sns
sns.set()
########################################

import numpy as np

import sys
sys.path.append("C:/Users/ayush/PycharmProjects/Black-Hole-X-ray-binary-Evolution/src")

from src.SpectralAnalysis import powerspectrum
from src.SpectralAnalysis.LightCurve import LightCurve

datadir = "data/"

data = np.loadtxt(datadir+"090122283_+071.87300_eventfile.dat")
print("Data shape : " + str(data.shape))


timestep = 0.001 ##the time resolution for the light curve
lc = LightCurve(data[:,0], timestep=timestep)

plt.figure()
plt.plot(lc.time, lc.counts)