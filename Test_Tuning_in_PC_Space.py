'''
Code to test, if we analyze the trajectory into PCs,
How is the neuron tuning to the submanifold (Linear or spherical) spanned by PCs
'''
import utils
from sklearn.decomposition import PCA
import  matplotlib.pylab as plt
from CNNScorer import NoIOCNNScorer
import numpy as np
#%%
exp_dir = r"D:\Generator_DB_Windows\data\with_CNN\caffe-net_fc8_0001\choleskycma_sgm3_uf10_cc0.00_cs0.05"
codes_all, generations = utils.codes_summary(exp_dir)

code_pca = PCA(n_components=50)
PC_Proj_codes = code_pca.fit_transform(codes_all)
PC_vectors = code_pca.components_

target_neuron = ('caffe-net', 'fc8', 1)
scorer = NoIOCNNScorer(target_neuron, exp_dir)
scorer.load_classifier()
#%%
img_list = GAN_interp_sphere_ang(PC_vectors[:3, :], sphere_norm=300, theta_ang_step= 180/10, phi_ang_step=180/10)
scores = scorer.test_score(img_list)
fig = visualize_img_list(img_list, scores)

#%%
img_list = GAN_interp_sphere_ang(PC_vectors[[0,48,49], :], sphere_norm=300, theta_ang_step= 180/10, phi_ang_step=180/10)
scores = scorer.test_score(img_list)
fig = visualize_img_list(img_list, scores)

#%%
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
phi = np.linspace(-np.pi/2, np.pi/2, 11)
theta = np.linspace(-np.pi/2, np.pi/2, 11)
phi, theta = np.meshgrid(phi, theta)

# The Cartesian coordinates of the unit sphere
x = np.cos(phi) * np.cos(theta)
y = np.cos(phi) * np.sin(theta)
z = np.sin(phi)
#%%
# Calculate the spherical harmonic Y(l,m) and normalize to [0,1]
fcolors = scores.reshape(x.shape)
fmax, fmin = fcolors.max(), fcolors.min()
fcolors = (fcolors - fmin)/(fmax - fmin)

# Set the aspect ratio to 1 so our sphere looks spherical
fig3d = plt.figure(figsize=plt.figaspect(1.))
ax = fig3d.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.seismic(fcolors))
# Turn off the axis planes
ax.set_axis_off()
plt.show()
#%%

phi = np.linspace(-np.pi/2, np.pi/2, 21)
theta = np.linspace(-np.pi/2, np.pi/2, 21)
img_list = GAN_interp_sphere_ang(PC_vectors[:3, :], sphere_norm=300, grid_shape=(21, 21),
                                 theta_ang_step= 180/20, phi_ang_step=180/20)
scores = scorer.test_score(img_list)
plt.pcolor(theta, phi, scores.reshape((21, 21)))
plt.colorbar()
plt.axis("image")
plt.show()
#%%
phi = np.linspace(-np.pi/2, np.pi/2, 21)
theta = np.linspace(-np.pi/2, np.pi/2, 21)
img_list = GAN_interp_sphere_ang(PC_vectors[[0,48,49], :], sphere_norm=300, grid_shape=(21, 21),
                                 theta_ang_step= 180/20, phi_ang_step=180/20)
scores = scorer.test_score(img_list)
plt.pcolor(theta, phi, scores.reshape((21, 21)))
plt.colorbar()
plt.axis("image")
plt.show()
