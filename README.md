## KMeansOps - PyKeops

This module implements three popular K-Means clustering algorithms: Lloyd's K-means algorithm, the Bisecting K-means algorithm, and mini-batch K-means algorithm. All algorithms are implemented using the PyTorch and PyKeops libraries to enable memory efficient computation on both CPUs and GPUs.

### Dependencies

This module requires the following libraries:

```bash
torch>=0.12.0", "pykeops>=2.1.2", "matplotlib>=3.7.1
```

### Installation

You can install the module by cloning this repository:

```
git clone https://github.com/SCALEDSL/biseckmeans-keops
python setup.py install
```

### Usage

Here is a basic example of how to use this simple module for Bisecting K-Means:

```python
from biseckmeanskeops import Clustering

# Check if CUDA is available and set the device ID
# "cuda:0" is used if CUDA is available, otherwise use the CPU
# Set the datatype to float32 if CUDA is available (for better performance), otherwise use float64
use_cuda = torch.cuda.is_available()
device_id = "cuda:0" if use_cuda else "cpu"
dtype = torch.float32 if use_cuda else torch.float64

# Define the data dimensions and cluster count
# N is the number of DATA_POINTS
# D is the dim[DATA_POINTS]
# K is the n[CLUSTERS]
N, D, K = 10000, 2, 50

# Generate a tensor of size NxD with random numbers
# Example tensor is initialized with values from a normal distribution scaled by 0.7 and shifted by 0.3
x = 0.7 * torch.randn(N, D, dtype=dtype, device=device_id) + 0.3

# Perform the bisecting K-means clustering
# cl contains the cluster ID for each data point
# c contains the centroids of the clusters
cl, c = Clustering.BisectingKMeans(x, K, device=device_id)

# Optional plotting for the clusters using the data points and centroids
Clustering.plot_clusters(x, cl, c)
```

Here is a basic example of how to use this simple module for Mini-batch K-Means:

```python
from kmeansops import Clustering

# Check if CUDA is available and set the device ID
# "cuda:0" is used if CUDA is available, otherwise use the CPU
# Set the datatype to float32 if CUDA is available (for better performance), otherwise use float64
use_cuda = torch.cuda.is_available()
device_id = "cuda:0" if use_cuda else "cpu"
dtype = torch.float32 if use_cuda else torch.float64

# Define the data dimensions and cluster count
# N is the number of DATA_POINTS
# D is the dim[DATA_POINTS]
# K is the n[CLUSTERS]
N, D, K = 10000, 2, 50

# Generate a tensor of size NxD with random numbers
# Example tensor is initialized with values from a normal distribution scaled by 0.7 and shifted by 0.3
x = 0.7 * torch.randn(N, D, dtype=dtype, device=device_id) + 0.3

# Define batch size and reassignment ratio for MiniBatch K-means
batch_size = 1024
reassignment_ratio = 0.01

# Perform the MiniBatch K-means clustering
# cl contains the cluster ID for each data point
# c contains the centroids of the clusters
cl, c = Clustering.MiniBatchKMeans(
    x, K, batch_size=batch_size, device=device_id, reassignment_ratio=reassignment_ratio
)

# Optional plotting for the clusters using the data points and centroids
Clustering.plot_clusters(x, cl, c)
```