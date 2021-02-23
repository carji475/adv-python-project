import numpy as np
from numpy import zeros

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






