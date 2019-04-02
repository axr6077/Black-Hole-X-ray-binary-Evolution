import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
import math
import sys

import scipy
import scipy.optimize
from scipy.stats.mstats import mquantiles as quantiles
import scipy.stats
import src.SpectralAnalysis.utils as utils
import src.SpectralAnalysis.powerspectrum as powerspectrum

try:
    import emcee
#    import acor
    emcee_import = True
except ImportError:
    print("Emcee and Acor not installed. Using Metropolis-Hastings algorithm for Markov Chain Monte Carlo simulations.")
    emcee_import = False

try:
    from numpy.random import choice
except ImportError:
    choice = utils.choice_hack

class MarkovChainMonteCarlo(object):
    """
    Markov Chain Monte Carlo for Bayesian QPO searches.
    Either wraps around emcee, or uses the
    Metropolis-Hastings sampler defined in this file.
    """
    def __init__(self, x, y, lpost, topt, tcov,
                 covfactor=1.0,
                 niter=5000,
                 nchain=10,
                 discard=None,
                 parname = None,
                 check_conv = True,
                 namestr='test',
                 use_emcee=True,
                 plot=True,
                 printobj = None,
                 m=1):


        self.m = m

        self.x = x
        self.y = y

        self.plot = plot
        print("<--- self.ps len MCMC: " + str(len(self.x)))
        self.topt = topt
        print("mcobs topt: " + str(self.topt))
        self.tcov = tcov*covfactor
        print("mcobs tcov: " + str(self.tcov))

        self.niter = niter
        self.nchain = nchain
        self.terr = np.sqrt(np.diag(tcov))
        self.lpost = lpost

        if discard == None:
            discard = math.floor(niter/2.0)

        mcall = []

        if emcee_import == False:
            print("Emcee not installed. Enforcing M-H algorithm!")
            use_emcee = False

        if use_emcee:

            nwalkers = self.nchain

            ndim = len(self.topt)

            p0 = [np.random.multivariate_normal(self.topt,self.tcov) for i in range(nwalkers)]

            sampler = emcee.EnsembleSampler(nwalkers,ndim, lpost, args=[False])

            pos, prob, state = sampler.run_mcmc(p0, 200)
            sampler.reset()

            sampler.run_mcmc(pos, niter, rstate0=state)

            mcall = sampler.flatchain

            print("The ensemble acceptance rate is: " + str(np.mean(sampler.acceptance_fraction)))
            self.L = np.mean(sampler.acceptance_fraction)*len(mcall)
            self.acceptance = np.mean(sampler.acceptance_fraction)
            try:
                self.acor = sampler.acor
                print("The autocorrelation times are: " +  str(sampler.acor))
            except ImportError:
                print("You can install acor: http://github.com/dfm/acor")
                self.acor = None
            except RuntimeError:
                print("D was negative. No clue why that's the case! Not computing autocorrelation time ...")
                self.acor = None
            except:
                print("Autocorrelation time calculation failed due to an unknown error: " + str(sys.exc_info()[0]) + ". Not computing autocorrelation time.")
                self.acor = None
        else:
            for i in range(nchain):

                #t0 = topt + choice([2.0, 3.0, -3.0, -2.0], size=len(topt))*self.terr

                mcout = MetropolisHastings(topt, tcov, lpost, niter = niter, parname = parname, discard = discard)

                mcout.create_chain(self.x, self.y)

                mcout.run_diagnostics(namestr = namestr +"_c"+str(i), parname=parname)

                mcall.extend(mcout.theta)

            self.L = mcout.L
        mcall = np.array(mcall)

        if check_conv == True:
            self.check_convergence(mcall, namestr, printobj = printobj)

        self.mcall = mcall.transpose()
        self.mcmc_infer(namestr=namestr, printobj = printobj)


    def check_convergence(self, mcall, namestr, printobj=None, use_emcee = True):

        rh = self._rhat(mcall, printobj)
        self.rhat = rh

        plt.scatter(rh, np.arange(len(rh))+1.0 )
        plt.axis([0.1,2,0.5,0.5+len(rh)])
        plt.xlabel("$R_hat$")
        plt.ylabel("Parameter")
        plt.title('Rhat')
        plt.savefig(namestr + '_rhat.png', format='png')
        plt.close()


        ci0, ci1 = self._quantiles(mcall)


        colours_basic = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        cneeded = int(math.ceil(len(ci0[0])/7.0))
        colours = []
        for x in range(cneeded):
            colours.extend(colours_basic)

        if self.plot:
            plt.plot(0,0)
            plt.axis([-2, 2, 0.5, 0.5+len(ci0)])
            for j in range(self.nchain):
                plt.hlines(y=[m+(j)/(4.0*self.nchain) for m in range(len(ci0))], xmin=[x[j] for x in ci0], xmax=[x[j] for x in ci1], color=colours[j])
            #plt.hlines(y=[m+1.0+(1)/(4*self.nchain) for m in np.arange(len(ci0))], xmin=[x[1] for x in ci0], xmax=[x[1] for x in ci1], color=colours[j])

            plt.xlabel("80% region (scaled)")
            plt.ylabel("Parameter")
            plt.title("80% quantiles")
            plt.savefig(namestr + "_quantiles.png", format="png")
            plt.close()

    def _rhat(self, mcall, printobj = None):

        #if printobj:
        #    print = printobj
        #else:
        #    from __builtin__ import print as print

        print("Computing Rhat. The closer to 1, the better!")

        rh = []

        for i,k in enumerate(self.topt):

            tpar = np.array([t[i] for t in mcall])

            tpar = np.reshape(tpar, (self.nchain, len(tpar)/self.nchain))

            sj = map(lambda y: np.var(y), tpar)
            W = np.mean(sj)

            mj = map(lambda y: np.mean(y), tpar)

            B = np.var(mj)*self.L


            mpv = ((float(self.L)-1.0)/float(self.L))*W + B/float(self.L)

            rh.append(np.sqrt(mpv/W))

            print("The Rhat value for parameter " + str(i) + " is: " + str(rh[i]) + ".")

            if rh[i] > 1.2:
                print("*** HIGH Rhat! Check results! ***")
            else:
                print("Good Rhat. Hoorah!")

        return rh

    def _quantiles(self, mcall):

        ci0, ci1 = [], []

        for i,k in enumerate(self.topt):

            print("I am in parameter: " + str(i))

            tpar = np.array([t[i] for t in mcall])

            tpar = np.reshape(tpar, (self.nchain, len(tpar)/self.nchain))

            intv = map(lambda y: quantiles(y, prob=[0.1, 0.9]), tpar)

            c0 = np.array([x[0] for x in intv])
            c1 = np.array([x[1] for x in intv])


            scale = np.mean(c1-c0)/2.0


            mt = map(lambda y: np.mean(y), tpar)

            offset = np.mean(mt)

            ci0.append((c0 - offset)/scale)
            ci1.append((c1 - offset)/scale)

        return ci0, ci1


    def mcmc_infer(self, namestr='test', printobj = None):

        covsim = np.cov(self.mcall)

        print("Covariance matrix (after simulations): \n")
        print(str(covsim))


        self.mean = map(lambda y: np.mean(y), self.mcall)
        self.std = map(lambda y: np.std(y), self.mcall)
        self.ci = map(lambda y: quantiles(y, prob=[0.05, 0.95]), self.mcall)

        print("-- Posterior Summary of Parameters: \n")
        print("parameter \t mean \t\t sd \t\t 5% \t\t 95% \n")
        print("---------------------------------------------\n")
        for i in range(len(self.topt)):
            print("theta[" + str(i) + "] \t " + str(self.mean[i]) + "\t" + str(self.std[i]) + "\t" + str(self.ci[i][0]) + "\t" + str(self.ci[i][1]) + "\n" )


        N = len(self.topt)
        print("N: " + str(N))
        n, bins, patches = [], [], []

        if self.plot:
            fig = plt.figure(figsize=(15,15))
            plt.subplots_adjust(top=0.925, bottom=0.025, left=0.025, right=0.975, wspace=0.2, hspace=0.2)
            for i in range(N):
                for j in range(N):
                    xmin, xmax = self.mcall[j][:1000].min(), self.mcall[j][:1000].max()
                    ymin, ymax = self.mcall[i][:1000].min(), self.mcall[i][:1000].max()
                    ax = fig.add_subplot(N,N,i*N+j+1)
                    ax.xaxis.set_major_locator(MaxNLocator(5))
                    ax.ticklabel_format(style="sci", scilimits=(-2,2))

                    if i == j:
                        #pass
                        ntemp, binstemp, patchestemp = ax.hist(self.mcall[i][:1000], 30, normed=True, histtype='stepfilled')
                        n.append(ntemp)
                        bins.append(binstemp)
                        patches.append(patchestemp)
                        ax.axis([ymin, ymax, 0, max(ntemp)*1.2])

                    else:

                        ax.axis([xmin, xmax, ymin, ymax])

                        ax.scatter(self.mcall[j][:1000], self.mcall[i][:1000], s=7)
                        xmin, xmax = self.mcall[j][:1000].min(), self.mcall[j][:1000].max()
                        ymin, ymax = self.mcall[i][:1000].min(), self.mcall[i][:1000].max()

                        try:
                            X,Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                            positions = np.vstack([X.ravel(), Y.ravel()])
                            values = np.vstack([self.mcall[j][:1000], self.mcall[i][:1000]])
                            kernel = scipy.stats.gaussian_kde(values)
                            Z = np.reshape(kernel(positions).T, X.shape)

                            ax.contour(X,Y,Z,7)
                        except ValueError:
                            print("Not making contours.")

            plt.savefig(namestr + "_scatter.png", format='png')
            plt.close()
        return


    def simulate_periodogram(self, nsim=5000):
        """
        Simulate periodograms from posterior samples of the
        broadband noise model.
        This method uses the results of an MCMC run to
        pick samples from the posterior and use the function
        stored in self.lpost.func to create a power spectral form.
        In order to transform this into a model periodogram,
        it picks for each frequency from an exponential distribution
        with a shape parameter corresponding to the model power
        at that frequency.
        """

        func = self.lpost.func

        nsim = min(nsim,len(self.mcall[0]))

        theta = np.transpose(self.mcall)
        #print "theta: " + str(len(theta))
        np.random.shuffle(theta)

        fps = []
        percount = 1.0

        for x in range(nsim):

            ain = theta[x]
            mpower = func(self.x, *ain)

            if self.m == 1:
                #print("m = 1")
                noise = np.random.exponential(size=len(self.x))
            else:
                #print("m = " + str(self.m))
                noise = np.random.chisquare(2*self.m, size=len(self.x))/(2.0*self.m)

            mpower = mpower*noise

            mps = powerspectrum.PowerSpectrum()
            mps.freq = self.x
            mps.ps = mpower
            mps.df = self.x[1] - self.x[0]
            mps.n = 2.0*len(self.x)
            mps.nphots = mpower[0]
            mps.m = self.m

            fps.append(mps)

        return np.array(fps)

#
class MetropolisHastings(object):

    def __init__(self, topt, tcov, lpost, niter = 5000,
                 parname=None, discard=None):

        self.niter = niter
        self.topt = topt
        self.tcov = tcov
        self.terr = np.sqrt(np.diag(tcov))
        self.t0 = topt + choice([2.0, 3.0, -3.0, -2.0], size=len(topt))*self.terr

        self.lpost = lpost
        self.terr = np.sqrt(np.diag(tcov))
        if discard == None:
            self.discard = int(niter/2)
        else:
            self.discard = int(discard)
        if parname == None:
            self.parname = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'iota', 'lappa', 'lambda', 'mu']
        else:
            self.parname = parname

    def create_chain(self, x, y, topt=None, tcov = None, t0 = None, dist='mvn'):

        if not topt == None:
            self.topt = topt
        if not tcov == None:
            self.tcov = tcov
        if not t0 == None:
            self.t0 = t0

        if dist=='mvn':
             dist = np.random.multivariate_normal

        accept = 0.0

        ### set up array
        ttemp, logp = [], []
        ttemp.append(self.t0)
        #lpost = posterior.PerPosterior(self.ps, self.func)
        logp.append(self.lpost(self.t0, neg=False))

        for t in np.arange(self.niter-1)+1:

            tprop = dist(ttemp[t-1], self.tcov)

            pprop = self.lpost(tprop)#, neg=False)

            logr = pprop - logp[t-1]
            logr = min(logr, 0.0)
            r= np.exp(logr)
            update = choice([True, False], size=1, p=[r, 1.0-r])

            if update:
                ttemp.append(tprop)
                logp.append(pprop)
                if t > self.discard:
                     accept = accept + 1
            else:
                ttemp.append(ttemp[t-1])
                logp.append(logp[t-1])

        self.theta = ttemp[self.discard+1:]
        self.logp = logp[self.discard+1:]
        self.L = self.niter - self.discard
        self.accept = accept/self.L
        return

    def run_diagnostics(self, namestr=None, parname=None, printobj = None):
        print("Markov Chain acceptance rate: " + str(self.accept) +".")

        if namestr == None:
            print("No file name string given for printing. Setting to 'test' ...")
            namestr = 'test'

        if parname == None:
           parname = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'iota', 'lappa', 'lambda', 'mu']

        fig = plt.figure(figsize=(12,10))
        adj =plt.subplots_adjust(hspace=0.4, wspace=0.4)

        for i,th in enumerate(self.theta[0]):
            ts = np.array([t[i] for t in self.theta])

            p1 = plt.subplot(len(self.topt), 3, (i*3)+1)
            p1 = plt.plot(ts)
            plt.axis([0, len(ts), min(ts), max(ts)])
            plt.xlabel("Number of draws")
            plt.ylabel("parameter value")
            plt.title("Time series for parameter " + str(parname[i]) + ".")

            p2 = plt.subplot(len(self.topt), 3, (i*3)+2)

            ### plotting histogram
            p2 = count, bins, ignored = plt.hist(ts, bins=10, normed=True)
            bnew = np.arange(bins[0], bins[-1], (bins[-1]-bins[0])/100.0)
            p2 = plt.plot(bnew, 1.0/(self.terr[i]*np.sqrt(2*np.pi))*np.exp(-(bnew - self.topt[i])**2.0/(2.0*self.terr[i]**2.0)), linewidth=2, color='r')
            plt.xlabel('value of ' + str(parname[i]))
            plt.ylabel('probability')
            plt.title("Histogram for parameter " + str(parname[i]) + ".")

            nlags = 30

            p3 = plt.subplot(len(self.topt), 3, (i*3)+3)
            acorr = utils.autocorr(ts,nlags=nlags, norm=True)
            p3 = plt.vlines(range(nlags), np.zeros(nlags), acorr, colors='black', linestyles='solid')
            plt.axis([0.0, nlags, 0.0, 1.0])

        plt.savefig(namestr  + "_diag.png", format='png',orientation='landscape')
        plt.close()