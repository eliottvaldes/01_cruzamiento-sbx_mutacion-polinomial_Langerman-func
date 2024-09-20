# import libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

# import langermann function
from calculate_aptitude import langermann_function


"""
@NOTE: This file does not have any direct relation with the Algorithm itself. It is just a helper function to visualize the Langermann function
@brief: Function to plot the Langermann function in 3D. Its just a helper function to visualize the function
@context: This function plots the Langermann function in 3D using matplotlib
@param zoom_factor: float -> Factor to zoom in the plot. Default value is 0.95
"""
def plot_langermann_function(zoom_factor=0.95):
    # create the meshgrid
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    
    # calculate the Z values
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = langermann_function([X[i, j], Y[i, j]])
    
    # plot the 3D surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # plot surface with color map
    surface = ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='none', alpha=0.9)

    # add a color bar for better visualization
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

    # add shadow (contour) in the bottom of the plot
    ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z) - 2, cmap=cm.viridis)
    
    # labels and title
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Z (Langermann)')
    ax.set_title('3D Plot of Langermann Function')

    # set viewing angle
    ax.view_init(elev=30, azim=120)

    # Set zoom: by adjusting the limits on x, y, and z axes
    x_mid = 5
    y_mid = 5
    z_min = np.min(Z)
    z_max = np.max(Z)

    # Apply zoom factor (zoom_factor < 1 zooms in, zoom_factor > 1 zooms out)
    ax.set_xlim([x_mid - 5 * zoom_factor, x_mid + 5 * zoom_factor])
    ax.set_ylim([y_mid - 5 * zoom_factor, y_mid + 5 * zoom_factor])
    ax.set_zlim([z_min - 2 * zoom_factor, z_max + 2 * zoom_factor])

    # show the plot
    plt.show()

if __name__ == '__main__':
    plot_langermann_function()