import numpy as np
import math

def const(freq, a):
    return np.array([np.exp(a) for x in freq])

def pl(freq, a, b, c=None):
    res = -a*np.log(freq) + b
    if c:
        return (np.exp(res) + np.exp(c))
    else:
        return np.exp(res)

"""
Lorentzian Profile for quasi-periodic oscillation
"""
def qpo(freq, a, b, c, d=None):

    gamma = np.exp(a)
    norm = np.exp(b)
    nu0 = c

    alpha = norm*gamma/(math.pi*2.0)
    y = alpha/((freq - nu0)**2.0 + gamma**2.0)

    if d is not None:
        y = y + np.exp(d)

    return y

def make_lorentzians(x):
    for f in x:
        def create_my_func(f):
            def lorentz(x, a, b, e):
                result = qpo(x, a, b, f, e)
                return result

            return lorentz

        yield (create_my_func(f))

def plqpo(freq, plind, beta, noise,a, b, c, d=None):
    powerlaw = pl(freq, plind, beta, noise)
    quasiper = qpo(freq, a, b, c, d)
    return powerlaw + quasiper

def bplqpo(freq, lplind, beta, hplind, fbreak, noise, a, b, c, d=None):
    powerlaw = bpl(freq, lplind, beta, hplind, fbreak, noise)
    quasiper = qpo(freq, a, b, c, d)
    return powerlaw + quasiper

def bpl(freq, a, b, c, d, e=None):
    logz = (c - a) * (np.log(freq) - d)
    logqsum = sum(np.where(logz < -100, 1.0, 0.0))
    if logqsum > 0.0:
        logq = np.where(logz < -100, 1.0, logz)
    else:
        logq = logz
    logqsum = np.sum(np.where((-100 <= logz) & (logz <= 100.0), np.log(1.0 + np.exp(logz)), 0.0))
    if logqsum > 0.0:
        logqnew = np.where((-100 <= logz) & (logz <= 100.0), np.log(1.0 + np.exp(logz)), logq)
    else:
        logqnew = logq

    logy = -a * np.log(freq) - logqnew + b

    if e:
        y = np.exp(logy) + np.exp(e)
    else:
        y = np.exp(logy)
    return y

def combine_models(*funcs, **kwargs):

    # assert that keyword 'mode' is given in function call
    try:
        assert kwargs in 'mode'
    # if that's not true, catch Assertion error and manually set mode = 'add'
    except AssertionError:
        kwargs["mode"] = 'add'