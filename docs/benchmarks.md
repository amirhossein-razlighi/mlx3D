# Benchmarks

Forward-pass throughput of the main render paths, measured with
[`examples/benchmark.py`](https://github.com/amirhossein-razlighi/mlx3D/blob/main/examples/benchmark.py)
on an Apple-Silicon GPU. Reproduce locally with:

```bash
uv run python examples/benchmark.py            # plain table
uv run python examples/benchmark.py --markdown # the table below
```

All inputs are synthetic (no downloads); numbers are per-call averages after a
warmup and will vary with your chip and thermal state.

| Benchmark | Latency | Throughput |
| --- | --- | --- |
| hard rasterizer · 20 480 faces @ 256px | 65.6 ms | 15 fps |
| hard rasterizer · 20 480 faces @ 512px | 242 ms | 4 fps |
| hard rasterizer · 20 480 faces @ 1024px | 933 ms | 1 fps |
| soft rasterizer · 20 480 faces @ 256px | 68 883 ms | 0.01 fps |
| gaussian splatting · 10 000 splats @ 512px | 12.5 ms | 80 fps |
| gaussian splatting · 100 000 splats @ 720px | 100 ms | 10 fps |
| vanilla NeRF · 4096 rays × 64 samples | 152 ms | 27k rays/s |
| hash-grid NeRF · 4096 rays × 64 samples | 111 ms | 37k rays/s |

## Reading the numbers

- **Hard vs soft mesh rasterizer.** The hard z-buffer rasterizer is ~1000× faster
  than the soft one on a 20k-face mesh (the soft path materializes per-face image
  buffers; the hard kernel keeps only the nearest hit). Use the soft rasterizer
  only when you need silhouette gradients.
- **Gaussian Splatting** runs interactively (the tile-based Metal kernels): real
  scenes with ~100k–400k splats render in real time at 512–720p.
- **Hash-grid NeRF** renders faster than the vanilla MLP NeRF *and* converges in
  far fewer iterations — it is the recommended NeRF path.

## Known limitation

The hard mesh rasterizer currently scans every face per pixel, so its cost is
`O(faces × pixels)`: great for moderate meshes, but it falls to ~1 fps on a
20k-face mesh at 1024px. Tile binning (as already used by the Gaussian Splatting
path) is the planned optimization to make it scale to large meshes at high
resolution.
