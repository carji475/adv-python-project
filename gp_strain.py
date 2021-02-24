import numpy as np
from scipy.optimize import minimize
from ctypes import *
from numpy.ctypeslib import ndpointer
import jax.scipy.linalg as jscl
import jax.numpy as jnp
from jax import jit, grad
from jax.config import config
config.update('jax_enable_x64', True)  # double precision is important here!


class covfunc():
    """
    Class representing the covariance function
    ...

    Attributes
    ----------
    covtype : str
        whether we're using squared exponential ('se') or matern ('matern')
    nu : float
        shape parameter (in case matern is used)

    """
    def __init__(self, covtype='matern', nu=2.5):
        """
        Parameters
        ----------
        covtype : str
            whether we're using squared exponential ('se') or matern ('matern')
            matern is default due to better numerical properties and for being
            more realistic
        nu : float
            shape parameter (in case matern is used)
        """
        super(covfunc, self).__init__()
        self.covtype = covtype
        self.nu = nu


def get_mperms(mx, my):
    """Returns basis function indices

    Parameters
    ----------
    mx, my : ints
        number of basis functions in x/y-directions

    Returns
    -------
    mperms : int
        (mx*my,2) array with all permutations: (1,1), (1,2), ..., (mx,my)
        that lie inside the ellipse mm1.^2/m_1^2+mm2.^2/m_2^2=1
    """
    mX, mY = np.meshgrid(np.arange(1,mx+1), np.arange(1,my+1))
    insideEllipse = np.where( (mX**2/mx**2 + mY**2/my**2 <= 1).flatten() )[0]
    return np.stack( (mX.reshape(-1)[insideEllipse],
                      mY.reshape(-1)[insideEllipse]), axis=1)


class gp_strain(object):
    """
    Class representing a strain reconstruction object. This class contains necessary
    problem inputs and methods needed to make the reconstruction. We use the
    Hilbert space basis function approximation by Solin/Särkkä, see
    (https://link.springer.com/article/10.1007/s11222-019-09886-w)
    ...

    Attributes
    ----------
    obs : ndarray
        (4,nobs+totAddSegs) array with entry and exit points for the measurements,
        each column having the form [x0; x1; y0; y1]. If the ray passes through
        multiple segments, place the columns left to right corresponding to
        first to last segment. Here 'nobs' is the number of measurements and
        'totAddSegs' is the total number of additional segments of all measurements
        (e.g. if a ray passes through 3 segments, 2 of them are considered additional)
    y : ndarray
        (nobs,) array with measured values
    pred : ndarray
        (P,2) array with coordinates for the prediction on the form [X(:) Y(:)]
    Lx, Ly : floats
        domain sizes in x/y-direction for the Laplace eigenvalue problem
    nrSegs - ndarray
        (nobs,) array specifying the number of segments the ray passes through for
        each measurement
    sigma_f : float
        signal variance hyperparameter
    l : float
        (square of) lengthscale hyperparameter
    sigma_n : float
        noise variance
    covfunc : covfunc
        covfunc object representing the covariance function
    mperms : ndarray
        (m,2) array with all permutations: (1,1), (1,2), ..., (mx,my)
        that lie inside the ellipse x^2/my^2+y^2/mx^2=1
    m : int
        number of basis functions
    addPrevSegs : ndarray
        (nobs,) vector specifying the total number of additional segments of all
        previous measurements
    A, B : floats
        parameters specifying the equilibrium constraint
    nobs : int
        number of measurements
    npred : int
        number of test points
    Phi : ndarray
        (nobs, m) array of basis functions evaluated at measurement inputs
    Phi_pred_T : ndarray
        (npred, m) array of basis functions evaluated at test inputs
    """

    def __init__(self, obs, y, pred, mx, my, Lx, Ly, nrSegs, sigma_f, l,
                 sigma_n, v, covfunc=covfunc()):
        """
        Parameters
        ----------
        obs : ndarray
            (4,N+totAddSegs) array with entry and exit points for the measurements,
            each column having the form [x0; x1; y0; y1]. If the ray passes through
            multiple segments, place the columns left to right corresponding to
            first to last segment. N is the number of measurements and totAddSegs is
            the total number of additional segments of all measurements (e.g. if a
            ray passes through 3 segments, 2 of them are considered additional)
        y : ndarray
            (N,) array with measured values
        pred : ndarray
            (P,2) array with coordinates for the prediction on the form [X(:) Y(:)]
        m_1, m_2 : ints
            number of basis functions in x/y-direction
        Lx, Ly : floats
            domain sizes in x/y-direction for the Laplace eigenvalue problem
        nrSegs - ndarray
            (N,) array specifying the number of segments the ray passes through for
            each measurement
        sigma_f : float
            signal variance hyperparameter
        l : float
            (square of) lengthscale hyperparameter
        sigma_n : float
            noise varaince hyperparameter
        v : float
            Poisson's ratio
        covfunc : covfunc
            covfunc object representing the covariance function
        """
        super(gp_strain, self).__init__()

        self.obs = obs
        self.y = y
        self.pred = pred
        self.Lx = Lx
        self.Ly = Ly
        self.nrSegs = nrSegs.astype(np.int32)
        self.sigma_f = sigma_f
        self.l = l
        self.sigma_n = sigma_n
        self.covfunc = covfunc

        # attributes derived from input parameters
        self.mperms = get_mperms(mx, my).astype(np.int32)
        self.m = self.mperms.shape[0]
        self.addPrevSegs = np.concatenate(
            (np.zeros(1), np.cumsum(nrSegs[:-1:]-1))).astype(np.int32)
        self.A = 1 + (1-v)/v
        self.B = - (self.A + 1)
        self.nobs = self.y.size
        self.npred = self.pred.shape[0]
        self.Phi = np.zeros((self.nobs, self.m))
        self.Phi_pred_T = np.zeros((3*self.npred, self.m))

        # fill the Phi-matrices -- we only need to compute the basis functions once
        self.build_phi()


    def build_phi(self):
        """Building the Phi-matrices. This is done in an external C-function,
        to which pointers are passed -- hence this function does not return
        anything.
        """

        # create a object representing the C-script
        cBN = CDLL('./build_phi.so')

        cBN.build_phi.restype = None  # no return

        # specify the data types passed to the function
        cBN.build_phi.argtypes = [c_int, c_int, c_int, ndpointer(c_int),
                                  ndpointer(c_double), ndpointer(c_double),
                                  c_double, c_double, c_double, c_double,
                                  ndpointer(c_int), ndpointer(c_int),
                                  ndpointer(c_double), ndpointer(c_double)]

        # call the C-function
        cBN.build_phi(self.nobs, self.npred, self.m, self.mperms,
                      self.obs, self.pred, self.Lx, self.Ly, self.A, self.B,
                      self.nrSegs, self.addPrevSegs, self.Phi, self.Phi_pred_T)

    def predict(self):
        """Computing the prediction mean and standard deviation

        Returns
        -------
        means : ndarray tuple
            tuple containing the mean components
        stds : ndarray tuple
            tuple containing the standard deviations
        """
        lambdam = self.getlambda()
        mean = self.Phi_pred_T@jscl.solve(
            self.Phi.T@self.Phi + np.diag(self.sigma_n/lambdam), self.Phi.T@self.y )
        std = np.sqrt(self.sigma_n*np.sum(self.Phi_pred_T*jscl.solve(
            self.Phi.T@self.Phi + np.diag(self.sigma_n/lambdam), self.Phi_pred_T.T ).T, 1))
        return (mean[::3], mean[1::3], mean[2::3]), (std[::3], std[1::3], std[2::3])

    def getlambda(self, hyperparams=None):
        """Computing the diagonal of the Lambda matrix (spectral values)
        We're using jax.numpy routines since this function is used in the optimisation

        Parameters
        ----------
        hyperparams : ndarray
            array with hyperparameters, if not passed the object attributes are used

        Returns
        -------
        lambda : ndarray
            (m,) array with spectral values
        """

        # angular frequencies
        omega = self.mperms*0.5*jnp.pi / jnp.array([[self.Lx, self.Ly]])

        if hyperparams is None:
            l = self.l
            sf = self.sigma_f
        else:  # when optimising, we must be able to provide the parameters directly
            l = hyperparams[1]
            sf = hyperparams[0]

        # compute the spectral density values
        if self.covfunc.covtype == 'matern':  # matern
            return sf * jnp.power( 2*self.covfunc.nu / l + jnp.sum(omega**2,1),
                                   -(self.covfunc.nu+1) ) / jnp.power(l, self.covfunc.nu)
        if self.covfunc.covtype == 'se':  # squared exponential
            return sf * l * jnp.exp( -0.5*l + jnp.sum(omega**2,1) )

    def optimise_ml(self, method='L-BFGS-B', options=None):
        """Updating hyperparameters by minimising the negative log likelihood.
        We are using the function 'minimize' from 'scipy.optimize'

        Parameters
        ----------
        method : str
            optimisation method, default 'L-BFGS-B'
        options : dict
            options to the selected optimisation method -- default as below
            (we should not use a mutable default value as an argument to a function)
        """
        if options is None:
            options = {'disp': True, 'maxiter': 10, 'iprint':1}

        # compile cost function using jit and create gradient handle
        cost_func = jit(self.nll)
        cost_grad = grad(cost_func)

        def npgradient(theta, *args):
            """Wrapper for scipy.optimize.minimize that returns a regular numpy
            array (instead of jax.numpy)

            Parameters
            ----------
            theta : ndarray
                optimisation parameters
            """
            return 0 + np.asarray(cost_grad(theta, *args))  # adding 0 since \
            # 'L-BFGS-B' otherwise complains about contig. problems ...

        # define start guess -- we're optimising the logarithm of the hyperparameters
        # to ensure positivity
        x0 = jnp.log(jnp.array([self.sigma_f, self.l,self.sigma_n]))

        # call the optimiser
        res = minimize(cost_func, x0, method=method, options=options, jac=npgradient)

        # update hyperparameters with optimisation result
        self.sigma_f = np.exp(res.x[0])
        self.l = np.exp(res.x[1])
        self.sigma_n = np.exp(res.x[2])

    def nll(self, loghyperparams):
        """Negative log likelihood

        Parameters
        ----------
        loghyperparams : ndarray
            array with logarithm of hyperparameters

        Returns
        -------
        nll : float
            negative log likelihood (up to a scale and offset)
        """
        hyperparams= jnp.exp(loghyperparams)
        lambdam = self.getlambda( hyperparams )
        sn = hyperparams[-1]

        # build the pd Z-matrix
        Z = self.Phi.T@self.Phi + jnp.diag(sn/lambdam)

        # use cholesky for numerical stability
        Zchol, low = jscl.cho_factor(Z)
        ZiPhiT = jscl.cho_solve((Zchol, low), self.Phi.T)

        # compute the log likelihood components
        logQ = (self.nobs-self.m)*loghyperparams[-1] + \
               2*jnp.sum(jnp.log(jnp.diag(Zchol))) + jnp.sum(jnp.log(lambdam))
        yTinvQy = 1/sn * self.y@( self.y -  self.Phi@(ZiPhiT@self.y) )

        return logQ + yTinvQy
