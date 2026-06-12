# Hash-Grid Encodings for NeRF

`HashGridEncoding` provides a compact trainable multi-resolution encoding for
NeRF-style fields. It follows the Instant-NGP idea: each level stores a small
hashed feature table, samples eight grid corners, trilinearly interpolates the
features, and concatenates levels.

```python
import mlx.core as mx
import mlx.nn as nn
from mlx3d.nn import HashGridEncoding

encoding = HashGridEncoding(
    num_levels=12,
    features_per_level=2,
    log2_hashmap_size=15,
    base_resolution=16,
    finest_resolution=512,
    bounds=(-1.0, 1.0),
)

points = mx.random.uniform(shape=(4096, 3), low=-1.0, high=1.0)
features = encoding(points)
print(features.shape)  # (4096, 24)
```

The encoder is an `mlx.nn.Module`; its hash tables are trainable parameters and
can be combined with a small MLP:

```python
class TinyField(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = HashGridEncoding(num_levels=8, features_per_level=2)
        self.mlp = nn.Sequential(
            nn.Linear(self.enc.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def __call__(self, x):
        return self.mlp(self.enc(x))
```

The current NeRF tutorial keeps the original sinusoidal NeRF architecture for
clarity. Use hash grids when you want a smaller field network and faster
spatial lookups for bounded scenes.
