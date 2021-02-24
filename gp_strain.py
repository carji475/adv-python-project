import numpy as np
from ctypes import *
from numpy.ctypeslib import ndpointer


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
    insideEllipse = np.where( (mX**2/mx**2 + mY**2/my**2 < 1).flatten() )[0]
    return np.stack( (mX.reshape(-1)[insideEllipse], mY.reshape(-1)[insideEllipse]), axis=1)


class gp_strain(object):
    """
    Class representing a strain reconstruction object. This class contains necessary
    problem inputs and methods needed to make the reconstruction. We use the
    Hilbert space basis function approximation by Solin/Särkkä, see
    (https://link.springer.com/article/10.1007/s11222-019-09886-w)
    GP_strainFieldRec(obs,y,pred,m_1,m_2,Lx,Ly,mu_att,nrSegs,addPrevSegs,sigma_f,M,sigma_n,E,v)
    ...

    Attributes
    ----------
    obs : float
        (4,nobs+totAddSegs) array with entry and exit points for the measurements,
        each column having the form [x0; x1; y0; y1]. If the ray passes through
        multiple segments, place the columns left to right corresponding to
        first to last segment. Here 'nobs' is the number of measurements and
        'totAddSegs' is the total number of additional segments of all measurements
        (e.g. if a ray passes through 3 segments, 2 of them are considered additional)
    y : float
        (nobs,) array with measured values
    pred : float
        (P,2) array with coordinates for the prediction on the form [X(:) Y(:)]
    mperms : int
        (m,2) array with all permutations: (1,1), (1,2), ..., (mx,my)
        that lie inside the ellipse x^2/my^2+y^2/mx^2=1
    m : int
        number of basis functions
    Lx, Ly : floats
        domain sizes in x/y-direction for the Laplace eigenvalue problem
    nrSegs - int
        (nobs,) array specifying the number of segments the ray passes through for
        each measurement
    sigma_f : float
        signal variance hyperparameter
    l : float
        lengthscale hyperparameter
    sigma_n : float
        noise std hyperparameter
    covfunc : covfunc
        covfunc object representing the covariance function
    addPrevSegs : int
        (nobs,) vector specifying the total number of additional segments of all
        previous measurements
    A, B : floats
        parameters specifying the equilibrium constraint
    nobs : int
        number of measurements
    npred : int
        number of test points
    Phi : double
        (nobs, m) array of basis functions evaluated at measurement inputs
    Phi_pred_T : double
        (npred, m) array of basis functions evaluated at test inputs
    """

    def __init__(self, obs, y, pred, mx, my, Lx, Ly, nrSegs, sigma_f, l, sigma_n, v, covfunc=covfunc()):
        """
        Parameters
        ----------
        obs : float
            (4,N+totAddSegs) array with entry and exit points for the measurements,
            each column having the form [x0; x1; y0; y1]. If the ray passes through
            multiple segments, place the columns left to right corresponding to
            first to last segment. N is the number of measurements and totAddSegs is
            the total number of additional segments of all measurements (e.g. if a
            ray passes through 3 segments, 2 of them are considered additional)
        y : float
            (N,) array with measured values
        pred : float
            (P,2) array with coordinates for the prediction on the form [X(:) Y(:)]
        m_1, m_2 : ints
            number of basis functions in x/y-direction
        Lx, Ly : floats
            domain sizes in x/y-direction for the Laplace eigenvalue problem
        nrSegs - int
            (N,) array specifying the number of segments the ray passes through for
            each measurement
        sigma_f : float
            signal variance hyperparameter
        l : float
            lengthscale hyperparameter
        sigma_n : float
            noise std hyperparameter
        v : float
            Poisson's ratio
        covfunc : covfunc
            covfunc object representing the covariance function
        """
        super(gp_strain, self).__init__()
        self.obs = obs
        self.y = y
        self.pred = pred
        self.mperms = get_mperms(mx, my).astype(np.int32)
        self.m = self.mperms.shape[0]
        self.Lx = Lx
        self.Ly = Ly
        self.nrSegs = nrSegs.astype(np.int32)
        self.sigma_f = np.log(sigma_f)
        self.l = np.log(l)
        self.sigma_n = np.log(sigma_n)
        self.covfunc = covfunc

        self.addPrevSegs = np.concatenate((np.zeros(1), np.cumsum(nrSegs[:-1:]-1))).astype(np.int32)
        self.A = 1 + (1-v)/v
        self.B = - (self.A + 1)
        self.nobs = self.y.size
        self.npred = self.pred.shape[0]
        self.Phi = np.zeros((self.nobs, self.m))
        self.Phi_pred_T = np.zeros((3*self.npred, self.m))
        self.build_phi() # fill the Phi-matrices


    def build_phi(self):
        """Building the Phi-matrices. This is done in an external C-function,
        to which pointers are passed -- hence this function does not return
        anything.
        """
        cBN = CDLL('./build_phi.so')
        cBN.build_phi.restype = None
        cBN.build_phi.argtypes = [c_int, c_int, c_int, ndpointer(c_int),
                                  ndpointer(c_double), ndpointer(c_double),
                                  c_double, c_double, c_double, c_double,
                                  ndpointer(c_int), ndpointer(c_int),
                                  ndpointer(c_double), ndpointer(c_double)]
        cBN.build_phi(self.nobs, self.npred, self.m, self.mperms,
                      self.obs, self.pred, self.Lx, self.Ly, self.A, self.B,
                      self.nrSegs, self.addPrevSegs, self.Phi, self.Phi_pred_T)

    def predict(self):
        """Computing the prediction mean and standard deviation

        Returns
        -------
        means : float
            tuple containing the mean components
        stds :
            tuple containing the standard deviations
        """
        lambdam = self.getlambda()
        expsn = np.exp(self.sigma_n)
        mean = self.Phi_pred_T@np.linalg.solve( self.Phi.T@self.Phi + np.diag(expsn/lambdam), self.Phi.T@self.y )
        std = expsn*np.sqrt(np.sum(self.Phi_pred_T*np.linalg.solve( self.Phi.T@self.Phi + np.diag(expsn/lambdam), self.Phi_pred_T.T ).T, 1))
        return (mean[::3], mean[1::3], mean[2::3]), (std[::3], std[1::3], std[2::3])

    def getlambda(self):
        """Computing the diagonal of the Lambda matrix (spectral values)

        Returns
        -------
        lambda : float
            (m,) array with spectral values
        """
        omega =  self.mperms*0.5*np.pi / np.array([[self.Lx, self.Ly]])
        expl = np.exp(self.l)
        expsf = np.exp(self.sigma_f)
        if self.covfunc.covtype=='matern':
            return expsf * np.power( 2*self.covfunc.nu / expl + np.sum(omega**2,1), -(self.covfunc.nu+1) ) / np.power(expl,self.covfunc.nu)
        if self.covfunc.covtype=='se':
            return expsf * expl * np.exp( -0.5*expl + np.sum(omega**2,1) )

#        def optimiseML():
            # selecting hyperpars by minimising nll

#        def nll():
            # optimisation cost function






