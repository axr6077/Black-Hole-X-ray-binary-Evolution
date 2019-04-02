import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.optimize
import scipy.stats
import scipy.signal
import copy
from src.SpectralAnalysis.parametricmodels import combine_models
try:
    from statsmodels.tools.numdiff import approx_hess
    comp_hessian = True
except ImportError:
    comp_hessian = False
from src.SpectralAnalysis import posterior
from src.SpectralAnalysis import powerspectrum

logmin = -100.0


class MaxLikelihood(object):
    def __init__(self, x, y, obs=True, fitmethod='powell'):

        self.x = x
        self.y = y
        self.obs = obs
        self.smooth3 = scipy.signal.wiener(self.y, 3)
        self.smooth5 = scipy.signal.wiener(self.y, 5)
        self.smooth11 = scipy.signal.wiener(self.y, 11)

        self._set_fitmethod(self, fitmethod)

    def _set_fitmethod(self, fitmethod):
        if fitmethod.lower() in ['simplex']:
            self.fitmethod = scipy.optimize.fmin
        elif fitmethod.lower() in ['powell']:
            self.fitmethod = scipy.optimize.fmin_powell
        elif fitmethod.lower() in ['gradient']:
            self.fitmethod = scipy.optimize.fmin_cg
        elif fitmethod.lower() in ['bfgs']:
            self.fitmethod = scipy.optimize.fmin_bfgs
        elif fitmethod.lower() in ['newton']:
            self.fitmethod = scipy.optimize.fmin_ncg
        elif fitmethod.lower() in ['leastsq']:
            self.fitmethod = scipy.optimize.leastsq
        elif fitmethod.lower() in ['constbfgs']:
            self.fitmethod = scipy.optimize.fmin_l_bfgs_b
        elif fitmethod.lower() in ['tnc']:
            self.fitmethod = scipy.optimize.fmin_tnc

        else:
            print("Minimization method not recognized. Using standard (Powell's) method.")
            self.fitmethod = scipy.optimize.fmin_powell

    def mlest(self, func, ain, obs=True, noise=None, neg=True, functype='posterior'):

        fitparams = self._fitting(func, ain, obs=True)

        if functype in ['p', 'post', 'posterior']:
            fitparams['deviance'] = 2.0 * func.loglikelihood(fitparams['popt'], neg=True)
        elif functype in ['l', 'like', 'likelihood']:
            fitparams['deviance'] = -2.0 * func(fitparams['popt'])

        print("Fitting statistics: ")
        print(" -- number of frequencies: " + str(len(self.x)))
        print(" -- Deviance [-2 log L] D = " + str(fitparams['deviance']))

        return fitparams

    def _fitting(self, optfunc, ain, optfuncprime=None, neg=True, obs=True):

        lenpower = float(len(self.y))

        if neg == True:
            if scipy.__version__ < "0.10.0":
                args = [neg]
            else:
                args = (neg,)
        else:
            args = ()

        funcval = 100.0
        while funcval == 100 or funcval == 200 or funcval == 0.0 or funcval == np.inf or funcval == -np.inf:
            if self.fitmethod == scipy.optimize.fmin_ncg:
                aopt = self.fitmethod(optfunc, ain, optfuncprime, disp=0, args=args)
            elif self.fitmethod == scipy.optimize.fmin_bfgs:
                aopt = self.fitmethod(optfunc, ain, disp=0, full_output=True, args=args)

                warnflag = aopt[6]
                if warnflag == 1:
                    print("*** ACHTUNG! Maximum number of iterations exceeded! ***")
                    # elif warnflag == 2:
                    # print("Gradient and/or function calls not changing!")
            else:
                aopt = self.fitmethod(optfunc, ain, disp=0, full_output=True, args=args)

            funcval = aopt[1]
            ain = np.array(ain) * ((np.random.rand(len(ain)) - 0.5) * 4.0)
        fitparams = {'popt': aopt[0], 'result': aopt[1]}

        fitparams['dof'] = lenpower - float(len(fitparams['popt']))
        fitparams['aic'] = fitparams['result'] + 2.0 * len(ain)
        fitparams['bic'] = fitparams['result'] + len(ain) * len(self.x)

        try:
            fitparams['deviance'] = 2.0 * optfunc.loglikelihood(fitparams['popt'])
        except AttributeError:
            fitparams['deviance'] = 2.0 * optfunc(fitparams['popt'])

        fitparams['sexp'] = 2.0 * len(self.x) * len(fitparams['popt'])
        fitparams['ssd'] = np.sqrt(2.0 * fitparams['sexp'])

        fitparams['smooth3'] = scipy.signal.wiener(self.y, 3)
        fitparams['smooth5'] = scipy.signal.wiener(self.y, 5)
        fitparams['smooth11'] = scipy.signal.wiener(self.y, 11)

        if obs == True:
            if self.fitmethod == scipy.optimize.fmin_bfgs:
                print("Approximating covariance from BFGS: ")
                covar = aopt[3]
                stderr = np.sqrt(np.diag(covar))

            else:
                print("Approximating Hessian with finite differences ...")
                if comp_hessian:
                    phess = approx_hess(aopt[0], optfunc, neg=args)

                    print("Hessian (empirical): " + str(phess))

                    covar = np.linalg.inv(phess)
                    stderr = np.sqrt(np.diag(covar))

                else:
                    print("Cannot compute hessian! Use BFGS or install statsmodels!")
                    covar = None
                    stderr = None

            print("Covariance (empirical): " + str(covar))

            fitparams['cov'] = covar
            fitparams['err'] = stderr
            print("The best-fit model parameters plus errors are:")
            for i, (x, y) in enumerate(zip(fitparams['popt'], stderr)):
                print("Parameter " + str(i) + ": " + str(x) + " +/- " + str(y))
            print("The Akaike Information Criterion of the power law model is: " + str(fitparams['aic']) + ".")

        return fitparams

    def compute_lrt(self, mod1, ain1, mod2, ain2, noise1=-1, noise2=-1, nmax=1):

        par1 = self.mlest(mod1, ain1, obs=self.obs, noise=noise1, nmax=nmax)
        par2 = self.mlest(mod2, ain2, obs=self.obs, noise=noise2, nmax=nmax)
        varname1 = "model1fit"
        varname2 = "model2fit"

        self.__setattr__(varname1, par1)
        self.__setattr__(varname2, par2)

        self.lrt = par1['deviance'] - par2['deviance']

        if self.obs == True:
            print("The Likelihood Ratio for models %s and %s is: LRT = %.4f" % (varname1, varname2, self.lrt))

        return self.lrt

    def __make_lorentzians(self, x):
        for f in x:
            def create_my_func(f):
                def lorentz(x, a, b, e):
                    result = powerspectrum.qpo(x, a, b, f, e)
                    return result

                return lorentz

            yield (create_my_func(f))

    def fitqpo(self, fitpars=None, residuals=False):

        if residuals:
            mfit = fitpars['mfit']
            residuals = np.array(fitpars["smooth5"]) / mfit

        else:
            residuals = np.array(fitpars["smooth5"])

        gamma_min = 2.0 * (self.x[2] - self.x[1])

        like_rat = []

        for f, func, res in zip(self.x[3:-3], self.__make_lorentzians(self.x[3:-3]), residuals[3:-3]):
            gamma_max = f / 2.0
            norm = np.mean(residuals) + np.var(residuals)
            ain = [gamma_min, norm, 0.0]
            pars = self.mlest(func, ain, noise=-1, obs=False, residuals=residuals)

            pars['fitfreq'] = f
            pars['residuals'] = residuals
            like_rat.append(pars)

        return like_rat

    def find_qpo(self, func, ain,
                 fitmethod='nlm',
                 plot=False,
                 plotname=None,
                 obs=False):

        optpars = self.mlest(func, ain, obs=obs, noise=-1)

        lrts = self.fitqpo(fitpars=optpars, residuals=True)

        like_rat = np.array([x['deviance'] for x in lrts])

        minind = np.where(like_rat == min(like_rat))
        minind = minind[0][0] + 3
        minfreq = self.x[minind]

        print("The frequency of the tentative QPO is: " + str(minfreq))

        residuals = self.smooth5 / optpars['mfit']

        best_lorentz = self.__make_lorentzians([minfreq])

        noiseind = len(optpars['popt']) - 1

        gamma_min = np.log((self.x[1] - self.x[0]) * 3.0)

        gamma_max = minfreq / 1.5

        print('combmod first component: ' + str(func))
        combmod = combine_models((func, len(optpars['popt'])), (powerspectrum.qpo, 3), mode='add')
        inpars = list(optpars['popt'].copy())
        inpars.extend(lrts[minind - 3]['popt'][:2])
        inpars.extend([minfreq])

        qpopars = self.mlest(combmod, inpars, obs=obs, noise=noiseind, smooth=0)

        lrt = optpars['deviance'] - qpopars['deviance']

        like_rat_norm = like_rat / np.mean(like_rat) * np.mean(self.y) * 100.0

        if plot:
            plt.figure()
            axL = plt.subplot(1, 1, 1)
            plt.plot(self.x, self.y, lw=3, c='navy')
            plt.plot(self.x, qpopars['mfit'], lw=3, c='MediumOrchid')
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel('Frequency')
            plt.ylabel('variance normalized power')

            axR = plt.twinx()
            axR.yaxis.tick_right()
            axR.yaxis.set_label_position("right")
            plt.plot(self.x[3:-3], like_rat, 'r--', lw=2, c="DeepSkyBlue")
            plt.ylabel("-2*log-likelihood")

            plt.axis([min(self.x), max(self.x), min(like_rat) - np.var(like_rat), max(like_rat) + np.var(like_rat)])

            plt.savefig(plotname + '.png', format='png')
            plt.close()

        return lrt, optpars, qpopars

    def plotfits(self, par1, par2=None, namestr='test', log=False):
        f = plt.figure(figsize=(12, 10))
        plt.subplots_adjust(hspace=0.0, wspace=0.4)
        s1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)

        if log:
            logx = np.log10(self.x)
            logy = np.log10(self.y)
            logpar1 = np.log10(par1['mfit'])
            logpar1s5 = np.log10(par1['smooth5'])

            p1, = plt.plot(logx, logy, color='black', linestyle='steps-mid')
            p1smooth = plt.plot(logx, logpar1s5, lw=3, color='orange')
            p2, = plt.plot(logx, logpar1, color='blue', lw=2)

        else:
            p1, = plt.plot(self.x, self.y, color='black', linestyle='steps-mid')
            p1smooth = plt.plot(self.x, par1['smooth5'], lw=3, color='orange')
            p2, = plt.plot(self.x, par1['mfit'], color='blue', lw=2)
        if par2:
            if log:
                logpar2 = np.log10(par2['mfit'])
                p3, = plt.plot(logx, logpar2, color='red', lw=2)
            else:
                p3, = plt.plot(self.x, par2['mfit'], color='red', lw=2)
            plt.legend([p1, p2, p3], ["observed periodogram", par1['model'] + " fit", par2['model'] + " fit"])
        else:
            plt.legend([p1, p2], ["observed periodogram", par1['model'] + " fit"])

        if log:
            plt.axis([min(logx), max(logx), min(logy) - 1.0, max(logy) + 1])
            plt.ylabel('log(Leahy-Normalized Power)', fontsize=18)

        else:
            plt.xscale("log")
            plt.yscale("log")

            plt.axis([min(self.x), max(self.x), min(self.y) / 10.0, max(self.y) * 10.0])
            plt.ylabel('Leahy-Normalized Power', fontsize=18)
        plt.title("Periodogram and fits for burst " + namestr, fontsize=18)

        s2 = plt.subplot2grid((4, 1), (2, 0), rowspan=1)
        pldif = self.y / par1['mfit']
        if par2:
            bpldif = self.y / par2['mfit']

        if log:
            plt.plot(logx, pldif, color='black', linestyle='steps-mid')
            plt.plot(logx, np.ones(len(self.x)), color='blue', lw=2)

        else:
            plt.plot(self.x, pldif, color='black', linestyle='steps-mid')
            plt.plot(self.x, np.ones(len(self.x)), color='blue', lw=2)
        plt.ylabel("Residuals, \n" + par1['model'] + " model", fontsize=18)

        if log:
            plt.axis([min(logx), max(logx), min(pldif), max(pldif)])

        else:
            plt.xscale("log")
            plt.yscale("log")
            plt.axis([min(self.x), max(self.x), min(pldif), max(pldif)])

        if par2:
            bpldif = self.y / par2['mfit']

            s3 = plt.subplot2grid((4, 1), (3, 0), rowspan=1)

            if log:
                plt.plot(logx, bpldif, color='black', linestyle='steps-mid')
                plt.plot(logx, np.ones(len(self.x)), color='red', lw=2)
                plt.axis([min(logx), max(logx), min(bpldif), max(bpldif)])

            else:
                plt.plot(self.x, bpldif, color='black', linestyle='steps-mid')
                plt.plot(self.x, np.ones(len(self.x)), color='red', lw=2)
                plt.xscale("log")
                plt.yscale("log")
                plt.axis([min(self.x), max(self.x), min(bpldif), max(bpldif)])

            plt.ylabel("Residuals, \n" + par2['model'] + " model", fontsize=18)

        ax = plt.gca()

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(14)

        if log:
            plt.xlabel("log(Frequency) [Hz]", fontsize=18)
        else:
            plt.xlabel("Frequency [Hz]", fontsize=18)

        plt.setp(s1.get_xticklabels(), visible=False)

        plt.savefig(namestr + '_ps_fit.png', format='png')
        plt.close()

        return


class PerMaxLike(MaxLikelihood):
    def __init__(self, ps, obs=True, fitmethod='powell'):
        # ps.freq = np.array(ps.freq[1:])
        self.x = ps.freq[1:]
        # ps.ps = np.array(ps.ps[1:])
        self.y = ps.ps[1:]

        self.ps = ps

        self.obs = obs
        self._set_fitmethod(fitmethod)

    def mlest(self, func, ain, obs=True, noise=None, nmax=1, residuals=None, smooth=0, m=1, map=True):

        if smooth == 0:
            power = self.y
        elif smooth == 3:
            power = self.smooth3
        elif smooth == 5:
            power = self.smooth5
        elif smooth == 11:
            power = self.smooth11
        else:
            raise Exception('No valid option for kwarg "smooth". Options are 0,3,5 and 11!')

        if not residuals is None:
            power = residuals

        lenpower = float(len(power))

        varobs = np.sum(power)
        varmod = np.sum(func(self.x, *ain))
        renorm = varobs / varmod

        if len(ain) > 1:
            ain[1] = ain[1] + np.log(renorm)
        if not noise is None:
            noisepower = power[-51:-1]
            meannoise = np.log(np.mean(noisepower))
            ain[noise] = meannoise

        pstemp = powerspectrum.PowerSpectrum()
        pstemp.freq = self.x
        pstemp.ps = power
        pstemp.df = self.ps.df

        if m == 1:
            lposterior = posterior.PerPosterior(pstemp, func)
        elif m > 1:
            lposterior = posterior.StackPerPosterior(pstemp, func, m)

        else:
            raise Exception("Number of power spectra is not a valid number!")

        if not map:
            lpost = lposterior.loglikelihood
        else:
            lpost = lposterior

        fitparams = self._fitting(lpost, ain, neg=True, obs=obs)

        fitparams["model"] = str(func).split()[1]
        fitparams["mfit"] = func(self.x, *fitparams['popt'])

        fitparams['merit'] = np.sum(((power - fitparams['mfit']) / fitparams['mfit']) ** 2.0)

        plrat = 2.0 * (self.y / fitparams['mfit'])
        # print(plrat)
        fitparams['sobs'] = np.sum(plrat)

        if nmax == 1:
            plmaxpow = max(plrat[1:])
            # print('plmaxpow: ' + str(plmaxpow))
            plmaxind = np.where(plrat == plmaxpow)[0]
            # print('plmaxind: ' + str(plmaxind))
            if len(plmaxind) > 1:
                plmaxind = plmaxind[0]
            elif len(plmaxind) == 0:
                plmaxind = -2
            plmaxfreq = self.x[plmaxind]

        else:

            plratsort = copy.copy(plrat)
            plratsort.sort()
            plmaxpow = plratsort[-nmax:]

            plmaxind, plmaxfreq = [], []
            for p in plmaxpow:
                try:
                    plmaxind_temp = np.where(plrat == p)[0]
                    if len(plmaxind_temp) > 1:
                        plmaxind_temp = plmaxind_temp[0]
                    elif len(plmaxind_temp) == 0:
                        plmaxind_temp = -2
                    plmaxind.append(plmaxind_temp)
                    plmaxfreq.append(self.x[plmaxind_temp])

                except TypeError:
                    plmaxind.append(None)
                    plmaxfreq.append(None)

        fitparams['maxpow'] = plmaxpow
        fitparams['maxind'] = plmaxind
        fitparams['maxfreq'] = plmaxfreq

        s3rat = 2.0 * (fitparams['smooth3'] / fitparams['mfit'])
        fitparams['s3max'] = max(s3rat[1:])
        try:
            s3maxind = np.where(s3rat == fitparams['s3max'])[0]
            if len(s3maxind) > 1:
                s3maxind = s3maxind[0]
            fitparams['s3maxfreq'] = self.x[s3maxind]
        except TypeError:
            fitparams["s3maxfreq"] = None
        s5rat = 2.0 * (fitparams['smooth5'] / fitparams['mfit'])
        fitparams['s5max'] = max(s5rat[1:])
        try:
            s5maxind = np.where(s5rat == fitparams['s5max'])[0]
            if len(s5maxind) > 1:
                s5maxind = s5maxind[0]
            fitparams['s5maxfreq'] = self.x[s5maxind]
        except TypeError:
            fitparams['s5maxfreq'] = None

        s11rat = 2.0 * (fitparams['smooth11'] / fitparams['mfit'])
        fitparams['s11max'] = max(s11rat[1:])
        try:
            s11maxind = np.where(s11rat == fitparams['s11max'])[0]
            if len(s11maxind) > 1:
                s11maxind = s11maxind[0]
            fitparams['s11maxfreq'] = self.x[s11maxind]
        except TypeError:
            fitparams['s11maxfreq'] = None

        df = (self.x[1] - self.x[0])
        bmax = int(self.x[-1] / (2.0 * (self.x[1] - self.x[0])))
        # print('bmax: ' + str(bmax))
        bins = [1, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 200, 300, 500]

        bindict = {}

        for b in bins:
            if b < bmax:
                if b == 1:
                    binps = self.ps
                else:
                    binps = self.ps.rebinps(b * df)
                binpsname = "bin" + str(b)
                bindict[binpsname] = binps
                binpl = func(binps.freq, *fitparams["popt"])
                binratio = 2.0 * np.array(binps.ps) / binpl
                maxind = np.where(binratio[1:] == max(binratio[1:]))[0]
                if len(maxind) > 1:
                    maxind = maxind[0]
                elif len(maxind) == 0:
                    maxind = -2
                binmaxpow = "bmax" + str(b)
                bindict[binmaxpow] = max(binratio[1:])
                binmaxfreq = "bmaxfreq" + str(b)
                bindict[binmaxfreq] = binps.freq[maxind + 1]
                bindict['binpl' + str(b)] = binpl

        fitparams["bindict"] = bindict
        plks = scipy.stats.kstest(plrat / 2.0, 'expon', N=len(plrat))
        fitparams['ksp'] = plks[1]

        if obs == True:
            print("The figure-of-merit function for this model is: " + str(
                fitparams['merit']) + " and the fit for " + str(fitparams['dof']) + " dof is " + str(
                fitparams['merit'] / fitparams['dof']) + ".")

            print("Fitting statistics: ")
            print(" -- number of frequencies: " + str(len(self.x)))
            print(" -- Deviance [-2 log L] D = " + str(fitparams['deviance']))
            print(" -- Highest data/model outlier 2I/S = " + str(fitparams['maxpow']))
            print("    at frequency f_max = " + str(fitparams['maxfreq']))

            print(" -- Highest smoothed data/model outlier for smoothing factor [3] 2I/S = " + str(fitparams['s3max']))
            print("    at frequency f_max = " + str(fitparams['s3maxfreq']))
            print(" -- Highest smoothed data/model outlier for smoothing factor [5] 2I/S = " + str(fitparams['s5max']))
            print("    at frequency f_max = " + str(fitparams['s5maxfreq']))
            print(
                " -- Highest smoothed data/model outlier for smoothing factor [11] 2I/S = " + str(fitparams['s11max']))
            print("    at frequency f_max = " + str(fitparams['s11maxfreq']))

            print(" -- Summed Residuals S = " + str(fitparams['sobs']))
            print(" -- Expected S ~ " + str(fitparams['sexp']) + " +- " + str(fitparams['ssd']))
            print(" -- KS test p-value (use with caution!) p = " + str(fitparams['ksp']))
            print(" -- merit function (SSE) M = " + str(fitparams['merit']))

        return fitparams

    def compute_lrt(self, mod1, ain1, mod2, ain2, noise1=-1, noise2=-1, m=1, map=True, nmax=1):

        par1 = self.mlest(mod1, ain1, obs=self.obs, noise=noise1, m=m, map=map, nmax=nmax)
        par2 = self.mlest(mod2, ain2, obs=self.obs, noise=noise2, m=m, map=map, nmax=nmax)

        varname1 = "model1fit"
        varname2 = "model2fit"

        self.__setattr__(varname1, par1)
        self.__setattr__(varname2, par2)

        self.lrt = par1['deviance'] - par2['deviance']

        if self.obs == True:
            print("The Likelihood Ratio for models %s and %s is: LRT = %.4f" % (varname1, varname2, self.lrt))

        return self.lrt