import numpy as np

# visualize uncertainty ellipse
def plot_ellipse(center, Sigma, ax, color='blue', rescale_factor=1, legend=True):
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma/rescale_factor)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    # Generate the ellipsoid (ellipse) points
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])

    # Scale and rotate the unit circle to create the ellipse
    ellipse = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ circle

    # Plot
    x = ellipse[0, :]/50 + center[0]
    y = ellipse[1, :]/50 + center[1]
    if legend:
        ax.plot(x, y, color=color, linestyle='--', label='Uncertainty ellipse')
    else:
        ax.plot(x, y, color=color, linestyle='--')