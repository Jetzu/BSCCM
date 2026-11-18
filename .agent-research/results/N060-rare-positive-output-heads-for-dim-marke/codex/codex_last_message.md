Delivered the modified harness, heads module, `PATCH.diff`, and supporting files here.

[README.md](README.md) has exact commands; [report.md](report.md) covers recommendations and data caveats; [cpu_smoke.log](cpu_smoke.log) records validation.

All 12 CPU arm/direction smokes passed, including corrected two-part loss, resumes, and default parity. `evaluate()` is byte-identical. Repository untouched.

Estimated GPU budget: **7.2 GPU-hours**. GPU validation remains for integration.