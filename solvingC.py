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

pred = np.stack((X.flatten(), Y.flatten()), axis=1) # test points

# expand domain
Lx = Cx*r2
Ly = Cy*r2

# hyperparameters (start guesses)
sigma_f = 1.0
l = 1.0
sigma_n = 1.0

# create a strain object
strain_object = gp_strain.gp_strain(obs, y, pred, mx, my, Lx, Ly, nrSegs, sigma_f, l, sigma_n, v)

# optimise
strain_object.optimise_ml()

# predict
eps_est, std_est = strain_object.predict()


## plotting
def plot_boundary(ax):
    """Plotting the boundary of the C-shape
    """
    linecol = 'black'
    linewidth = 1.0
    theta = np.linspace(np.pi/4,7*np.pi/4,1000)
    ax.plot(r1*np.cos(theta), r1*np.sin(theta), color=linecol, linewidth=linewidth)
    ax.plot(r2*np.cos(theta), r2*np.sin(theta), color=linecol, linewidth=linewidth)
    ax.plot(np.cos(theta[0])*np.array([r1, r2]),\
            np.sin(theta[0])*np.array([r1, r2]), color=linecol, linewidth=linewidth)
    ax.plot(np.cos(theta[-1])*np.array([r1, r2]),\
            np.sin(theta[-1])*np.array([r1, r2]), color=linecol, linewidth=linewidth)


import cmocean
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from scipy.io import loadmat
fea_data = loadmat('./data/JPARC_fea.mat') # finite element solutions
fea_xx = LinearNDInterpolator(fea_data['pxx'], fea_data['vxx'])
fea_xy = LinearNDInterpolator(fea_data['pxy'], fea_data['vxy'])
fea_yy = LinearNDInterpolator(fea_data['pyy'], fea_data['vyy'])
feas = [fea_xx, fea_xy, fea_yy]

# caxis settings
vmin_eps = -1e-3
vmax_eps = 1e-3
vmin_std = 1e-5
vmax_std = 1e-4

# select colormap and shading
cmap = cmocean.cm.balance
shading = 'auto'

# strain component title names
eps_names = [r'$\epsilon_{xx}$', r'$\epsilon_{xy}$', r'$\epsilon_{yy}$']

# create figure and add plots
fig = plt.figure(figsize=(10,10))
figind=1
for qq in range(3):
    ax = fig.add_subplot(3,3, qq+1)
    ax.pcolormesh(X, Y, 1.15*feas[qq](pred).reshape((ny,nx)), shading=shading, cmap=cmap, vmin=vmin_eps,vmax=vmax_eps)
    ax.set_title(eps_names[qq], fontsize=20)
    if qq==0:
        ax.text(-0.2,0.5,'FEA', fontsize=15, rotation='vertical', va='center', transform=ax.transAxes)
    plot_boundary(ax)
    ax.axis('off')

    ax = fig.add_subplot(3,3, qq+4)
    ax_eps = ax.pcolormesh(X, Y, eps_est[qq].reshape((ny,nx)), shading=shading, cmap=cmap, vmin=vmin_eps,vmax=vmax_eps)
    if qq==0:
        ax.text(-0.2,0.5,'GP mean', fontsize=15, rotation='vertical', va='center', transform=ax.transAxes)
    plot_boundary(ax)
    ax.axis('off')

    ax = fig.add_subplot(3,3, qq+7)
    ax_std = ax.pcolormesh(X, Y, std_est[qq].reshape((ny,nx)), shading=shading, cmap=cmap, vmin=vmin_std,vmax=vmax_std)
    if qq==0:
        ax.text(-0.2,0.5,'Predicted std', fontsize=15, rotation='vertical', va='center', transform=ax.transAxes)
    plot_boundary(ax)
    ax.axis('off')


# add colorbars
plt.subplots_adjust(left=0.05, top=0.95, bottom=0.01, right=0.8)
cax_eps = plt.axes([0.85, 0.35, 0.075, 0.58])
cbar_eps = plt.colorbar(ax_eps, cax=cax_eps)
cbar_eps.set_ticks(np.arange(vmin_eps,vmax_eps+1e-4,5e-4) )
cax_std = plt.axes([0.85, 0.02, 0.075, 0.24])
cbar_std = plt.colorbar(ax_std, cax=cax_std)
cbar_std.set_ticks(np.arange(0,vmax_std+1e-5,25e-6) )

plt.show()

## save plot
fig.savefig('strain_plot.png', format='png', bbox_inches='tight')
