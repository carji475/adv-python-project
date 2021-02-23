import numpy as np

# covariance function object
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


# main GP class
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
    addPrevSegs : int
        (N,) vector specifying the total number of additional segments of all
        previous measurements TODO: derived
    sigma_f : float
        signal variance hyperparameter
    l : float
        lengthscale hyperparameter
    sigma_n : float
        noise std hyperparameter
    v - Poisson's ratio
    covfunc : covfunc
        covfunc object representing the covariance function


    """

    def __init__(self, obs, y, pred, mx, my, Lx, Ly, nrSegs, sigma_f, l, sigma_n, v, covfunc=covfunc()):
        super(gp_strain, self).__init__()
        # TODO: initialise all variables
        self.obs = obs
        self.y = y
        self.pred = pred
        self.mperms = self.get_mperms(mx, my)
        self.Lx = Lx
        self.Ly = Ly
        self.nrSegs = nrSegs
        self.sigma_f = sigma_f
        self.l = l
        self.sigma_n = sigma_n
        self.v = v
        self.covfunc = covfunc


        def get_mperms(self, mx, my):
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
            insideEllipse = np.where( mX**2/mx**2 + mY**2/my**2 <= 1 )[0]
            return np.stack((mX.reshape(-1)[insideEllipse], mY.reshape(-1)[insideEllipse]),axis=1)


#       def build_phi():
            # calling c-function

#        def optimiseML():
            # selecting hyperpars by minimising nll

#        def nll():
            # optimisation cost function






