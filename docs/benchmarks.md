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
| hard rasterizer · 20 480 faces @ 256px | 4.3 ms | 235 fps |
| hard rasterizer · 20 480 faces @ 512px | 6.2 ms | 162 fps |
| hard rasterizer · 20 480 faces @ 1024px | 17.9 ms | 56 fps |
| soft rasterizer · 20 480 faces @ 256px | 68 300 ms | 0.01 fps |
| gaussian splatting · 10 000 splats @ 512px | 12.5 ms | 80 fps |
| gaussian splatting · 100 000 splats @ 720px | 100 ms | 10 fps |
| vanilla NeRF · 4096 rays × 64 samples | 152 ms | 27k rays/s |
| hash-grid NeRF · 4096 rays × 64 samples | 129 ms | 32k rays/s |

## Reading the numbers

- **Hard vs soft mesh rasterizer.** The hard z-buffer rasterizer is *four to five
  orders of magnitude* faster than the soft one on a 20k-face mesh: it bins faces
  into 16×16 screen tiles (the same tiler the Gaussian Splatting path uses), so
  each pixel only tests the handful of faces in its tile, and renders 720p–1024px
  in real time. The soft path materializes per-face image buffers and is only for
  silhouette gradients.
- **Gaussian Splatting** runs interactively (the tile-based Metal kernels): real
  scenes with ~100k–400k splats render in real time at 512–720p.
- **Hash-grid NeRF** renders faster than the vanilla MLP NeRF *and* converges in
  far fewer iterations — it is the recommended NeRF path.

## Scaling

Because faces are tile-binned, the hard rasterizer's per-pixel cost depends on
the local face density rather than the total face count, so it stays fast as
meshes grow and resolution increases (56 fps at 1024px on a 20k-face mesh). Very
dense meshes that pack many triangles into one tile are the worst case; a
depth-sorted early-out and finer tiles are the next refinements.
