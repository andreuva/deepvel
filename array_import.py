import xarray as xr
import matplotlib.pyplot as plt

# Lazy get a single snapshot (you can use open_mfdataset() to get the timeseries)
data = xr.open_dataset("https://simulations.sdc.leibniz-kis.de/opendap/co5bold/SVGd3r05bn0cp1p/rhd_t00014400s.h5",decode_times=False)
data

# Plot the vertical component of velocity ("v3") at z ("xc3") index = 53
data["v3"].isel(xc3=53).plot()
plt.show()
