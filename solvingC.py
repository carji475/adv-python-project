import numpy as np
from scipy.io import loadmat
import gp_strain

data = loadmat('./data/JPARC_processed.mat')
y = data['y_meas'].astype(np.float64).flatten()
obs = np.ascontiguousarray(data['obs'].astype(np.float64))
nrSegs = data['nrSegs'].astype(np.int32).flatten()

# Approximation parameters
Cx = 2.5
Cy = 2.5

# number of basis functions
mx = 30
my = 30

# material parameters
v = 0.3 # poissons ratio

# test mesh
numr = 100
numtheta = 360
r1 = 3.5e-3
r2 = 10e-3
r, theta = np.meshgrid(np.linspace(r1,r2,numr), np.linspace(45,360-45,numtheta))
X = r*np.cos(theta*np.pi/180)
Y = r*np.sin(theta*np.pi/180)

nx = X.shape[1]
ny = X.shape[0]

pred = np.stack((X.flatten(), Y.flatten()), axis=1)

# expand domain
Lx = Cx*r2
Ly = Cy*r2

# hyperparameters (start guesses)
sigma_f = 1.0
l = 1.0
sigma_n = 1.0

# create a strain object
strain_object = gp_strain.gp_strain(obs, y, pred, mx, my, Lx, Ly, nrSegs, sigma_f, l, sigma_n, v)
