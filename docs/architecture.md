# Architecture

```text
Night/Fog Image
      |
      v
  CNN Encoder -------- local spatial features
      |
      v
 Channel Attention ---- feature weighting / glare-sensitive emphasis
      |
      v
 Transformer Block ----- global context
      |
      v
 Upsampling + Skip ----- feature fusion
      |
      v
 Enhanced Night Image
      |
      +---- PSNR / SSIM (when a clean reference exists)
```

## Prototype design

- **CNN encoder:** learns local texture and edge features.
- **Channel attention:** adaptively reweights feature channels.
- **Transformer block:** models long-range spatial relationships.
- **Skip connection:** preserves useful low-level image details.
- **Residual output:** predicts an enhancement correction while retaining the original image.

The implementation is intentionally compact so it can run on a normal laptop. It should be treated as a prototype architecture for demonstration and experimentation, not as a production restoration model.
