PyCUDAML
===

Some machine learning kernels write in CUDA with python binding.

KMeans
---

Implements K-Means clustering algorithm with CUDA, and support Numpy interface.
```python
import numpy as np
from PyCUDAML.KMeans import KMeans

# Data matrix as a numpy array.
X = np.random.rand(60000, 748)

k = KMeans(num_clusters=16, threshold=1e-2,
           seed=1117, max_iter=1000)

# Returned cluster centers are also numpy array.
cluster_centers = k.fit(X)
```

Installation
===
You need to install [CUDA](http://www.nvidia.com/object/cuda_home_new.html) first.

```bash
python setup.py install
```
