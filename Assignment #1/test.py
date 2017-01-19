import error_ellipse as ellipse
import numpy as np
import matplotlib.pyplot as plt
import math

points = np.random.multivariate_normal(
        mean=(3.2, 5.1),
        cov=[[1.2, -0.5],[-0.5, 3.3]], size=1000)
# Plot the raw points...
x, y = points.T
plt.plot(x, y, 'b.')
# Plot a transparent 3 standard deviation covariance ellipse
ellipse.plot_point_cov(points, nstd=math.sqrt(2), color='k', alpha=0.3)
plt.grid(True);
plt.axis('equal')
plt.show()
