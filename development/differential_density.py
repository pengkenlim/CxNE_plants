# %%
from sklearn.neighbors import KernelDensity
import numpy as np
# %%
# Sample data
n_dimensions = 2
data = np.random.rand(100, n_dimensions)  # 100 points in n-dimensional space

# Kernel density estimation
kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(data)
log_density = kde.score_samples(data)  # Log density for each point
density = np.exp(log_density)  # Convert log density to regular density
# %%
