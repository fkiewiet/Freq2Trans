Hi Laurent and Kees,

I want to restart the warm-start experiment in the cleanest form.

The version I think is most defensible is a one-shot warm start: compute the
low-frequency input with the same free-space Green's-function solver used in
training, apply the transfer model `T_{omega/2 -> omega}`, and then use that
prediction as the initial guess for the target PML solve with `GMRES`.

I think the right immediate metric is the quality of the initial guess on the
interior domain, since that directly tests whether the trained transfer is
working as intended. The longer-term solver goal is of course reduced GMRES
iteration count, but if the current preconditioned solve already converges in
only a few steps, then a real improvement in initial field quality may not show
up yet as a large iteration reduction.

So my plan is to report both:

1. interior field error at `k=0` for zero start versus warm start;
2. GMRES iterations and time to a fixed tolerance.

That should separate "is the warm start good?" from "does it help this
particular solver regime?"
