# Condition C: Delta-map update prompt.
# The model is given M_{t-1} and O_t, and must output a compact JSON delta
# describing only the objects whose states should change. The delta is applied
# programmatically to form M_t = Apply(M_{t-1}, delta_t).
