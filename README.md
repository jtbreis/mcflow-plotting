# mcflow-plotting

Code for plotting in mcflow style: fonts, palettes, reusable figure helpers, and **hot-wire / pitot analysis** (HDF5 IO, calibration, longitudinal spectrum, Taylor scales) under `mcflow_plotting.hotwire`.

Install (editable) using the **same Python environment** as Jupyter:

```bash
pip install -e ./mcflow-plotting
```

If imports fail with ``(unknown location)`` or missing names, the kernel is usually a different env—reinstall there, or add the source tree to the path:

```python
import sys
sys.path.insert(0, "/workspace/mcflow-plotting/src")  # adjust to your clone
```

Example imports:

```python
from mcflow_plotting import NU_AIR, load_calibrated_hotwire_velocity, set_font, FLOW_COLORS
from mcflow_plotting.hotwire import compute_normalized_spectrum_run
from mcflow_plotting.plots.hotwire import plot_tke_spectrum_kolmogorov_normalized
```
