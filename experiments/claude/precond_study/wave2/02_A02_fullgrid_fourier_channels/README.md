# 02_A02_fullgrid_fourier_channels

Question:
- Does adding Fourier positional channels help the full-grid model learn
  oscillatory transfer more effectively?

Status:
- scaffold only

Planned change:
- keep the full-grid `precond_v3` task
- add fixed Fourier feature channels to the model input
- keep optimizer and split unchanged

Why it is wave 2:
- promising representation change after the bottleneck probes
