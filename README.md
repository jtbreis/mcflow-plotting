# mcflow-plotting

Code for plotting in mcflow style: fonts, palettes, reusable figure helpers, and **hot-wire / pitot analysis** (HDF5 IO, calibration, longitudinal spectrum, Taylor scales) under `mcflow_plotting.hotwire`.

Install (editable) from the repo root (see the workspace **README** for GitHub Codespaces and `requirements-dev.txt`):

```bash
pip install -e ./mcflow-plotting
```

If imports fail with ``(unknown location)`` or missing names, the interpreter is usually a different env—reinstall there, or add the source tree to the path:

```python
import sys
sys.path.insert(0, "/workspaces/hotwire_data_processing/mcflow-plotting/src")  # Codespaces path; adjust locally
```

Example imports:

```python
from mcflow_plotting import NU_AIR, load_calibrated_hotwire_velocity, set_font, FLOW_COLORS
from mcflow_plotting.hotwire import compute_normalized_spectrum_run
from mcflow_plotting.plots.hotwire import plot_tke_spectrum_kolmogorov_normalized
```
