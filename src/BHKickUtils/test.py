from src.BHKickUtils import tools as gwt
from src.BHKickUtils import BHKick as bhk
import numpy as np

noi = bhk.noise()

sigma = 10
velocity = 0
tshift = 0
phase = 0

wave = bhk.waveform("cbc_q1.00_M60_d410_t140.00_p0.00.dat")

ifftwaveht = gwt.infft(wave.hf,1/wave.dT)
wavehf = gwt.fft(ifftwaveht,1/wave.dT)[1]

gwt.plot(wave.t,(wave.ht-ifftwaveht),title='Residual Plot - Success')

print('Overlap Calc = ' + str(gwt.overlap_calc(wavehf,wave.hf,wave.f,noi.aligochar)))

velowave = bhk.velowaveform(wave.t,wave.ht,velocity,sigma,tshift,phase)

veloht = gwt.infft(velowave.hf,1/velowave.dT)

gwt.plot(wave.t,(wave.ht-veloht),title='Residual Plot - SUCCESS')

gwt.wave_plot(wave.t,wave.ht,velowave.t,velowave.ht)

print('Overlap Calc = ' + str(gwt.overlap_calc(velowave.hf,wave.hf,wave.f,noi.aligochar)))

phase = 2*np.pi
velocity = 0
tshift =0

velowave = bhk.velowaveform(wave.t,wave.ht,velocity,sigma,tshift,phase)
veloht = gwt.infft(velowave.hf,1/velowave.dT)
gwt.plot(wave.t,(wave.ht-veloht),title='Residual Plot - Aligned')
print('Overlap Calc = ' + str(gwt.overlap_calc(velowave.hf,wave.hf,wave.f,noi.aligochar)))

phase = 0
velocity=0.01

velowave = bhk.velowaveform(wave.t,wave.ht,velocity,sigma,tshift,phase)
veloht = gwt.infft(velowave.hf,1/velowave.dT)

gwt.wave_plot(wave.t,wave.ht,velowave.t,velowave.ht)
print('(1-Overlap)*10^5 Calc = ' + str((1-gwt.overlap_calc(velowave.hf,wave.hf,wave.f,noi.aligochar))*10**5))

phase = 0
velocity=0.005

velowave = bhk.velowaveform(wave.t,wave.ht,velocity,sigma,tshift,phase)
veloht = gwt.infft(velowave.hf,1/velowave.dT)

gwt.wave_plot(wave.t,wave.ht,velowave.t,velowave.ht)

print('(1-Overlap)*10^5 Calc = ' + str((1-gwt.overlap_calc(velowave.hf,wave.hf,wave.f,noi.aligochar))*10**5))

sigma = 20
velolist = []
sigtwenlist = []
for velo in np.arange(-0.017, 0.018, 0.001):
    velolist.append(velo)
    velowave = bhk.velowaveform(wave.t, wave.ht, velo, sigma, tshift, phase)
    moval = (1 - gwt.overlap_calc(velowave.hf, wave.hf, wave.f, noi.aligochar)) * 10 ** 5
    sigtwenlist.append(moval)
    print('(1-Overlap)*10^5 Calc @ ' + str(velo) + 'c & o=' + str(sigma) + ': ' + str(moval))

sigma = 10
sigtenlist = []
for velo in np.arange(-0.017, 0.018, 0.001):
    velowave = bhk.velowaveform(wave.t, wave.ht, velo, sigma, tshift, phase)
    moval = (1 - gwt.overlap_calc(velowave.hf, wave.hf, wave.f, noi.aligochar)) * 10 ** 5
    sigtenlist.append(moval)
    print('(1-Overlap)*10^5 Calc @ ' + str(velo) + 'c & o=' + str(sigma) + ': '
          + str(moval))

gwt.oval_plot(velolist,sigtenlist,sigtwenlist)

tclist = []
ovallist = []
velocity = 0.01
sigma = 10
phase = 0
for tshift in np.arange(-0.001, 0.0011, 0.0002):
    tclist.append(tshift)
    velowave = bhk.velowaveform(wave.t, wave.ht, velocity, sigma, tshift, phase)
    oval = gwt.overlap_calc(velowave.hf, wave.hf, wave.f, noi.aligochar)
    ovallist.append(oval)
    print('Overlap Calc @ ' + str(tshift) + ' s ' + str(velocity) + 'c & o=' + str(sigma) + ': ' + str(oval))


gwt.tval_plot(tclist,ovallist)

plist = []
ovallist = []
velocity = 0.01
sigma = 10
tshift = 0
for phase in np.arange(-3.5, 3 + 1, 0.5):
    plist.append(phase)
    velowave = bhk.velowaveform(wave.t, wave.ht, velocity, sigma, tshift, phase)
    oval = gwt.overlap_calc(velowave.hf, wave.hf, wave.f, noi.aligochar)
    ovallist.append(oval)
    print('Overlap Calc @ ' + str(phase) + ' rad ' + str(velocity) + 'c & o=' + str(sigma) + ': ' + str(oval))

gwt.pval_plot(plist,ovallist)

plist = []
tlist = []
olist = []
velocity = 0.01
sigma = 10
for tshift in np.arange(-0.00004,-0.00003,0.000002):
    for phase in np.arange(-0.02,-0.01,0.002):
        plist.append(phase)
        tlist.append(tshift)
        velowave = bhk.velowaveform(wave.t,wave.ht,velocity,sigma,tshift,phase)
        oval = gwt.overlap_calc(velowave.hf,wave.hf,wave.f,noi.aligochar)
        olist.append(oval)
        print('Overlap Calc @ '+ str(phase) + ' rad '+ str(tshift) + ' s ' +str(velocity)
              + 'c & o=' + str(sigma) + ': ' + str(oval))

gwt.contour_plot(tlist,plist,olist)

plisttwen = []
tlisttwen = []
olisttwen = []
velocity = 0.01
sigma = 20
for tshift in np.arange(-0.002,0.0021,0.0005):
    for phase in np.arange(-0.2,0.21,0.05):
        plisttwen.append(phase)
        tlisttwen.append(tshift)
        velowave = bhk.velowaveform(wave.t,wave.ht,velocity,sigma,tshift,phase)
        oval = gwt.overlap_calc(velowave.hf,wave.hf,wave.f,noi.aligochar)
        olisttwen.append(oval)
        print('Overlap Calc @ '+ str(phase) + ' rad '+ str(tshift) + ' s ' +str(velocity)
              + 'c & o=' + str(sigma) + ': ' + str(oval))

gwt.contour_plot(tlisttwen,plisttwen,olisttwen)

phase = 0
velocity=-0.5
sigma = 40
velowave = bhk.velowaveform(wave.t,wave.ht,velocity,sigma,tshift,phase)
veloht = gwt.infft(velowave.hf,1/velowave.dT)
gwt.wave_plot(wave.t,wave.ht,velowave.t,velowave.ht)
print('(1-Overlap)*10^5 Calc = ' + str((1-gwt.overlap_calc(velowave.hf,wave.hf,wave.f,noi.aligochar))*10**5))