# Climate emulator handbook

The climate emulator exposes functions for quantitative temperature analysis.

- `query_lat_and_lon`: city name; returns the latitude and longitude used by the climate functions below.
- `history_temperature`: latitude (degrees, -90 to 90), longitude (degrees, -180 to 180), and year (1850-2014); returns annual mean temperature.
- `future_temperature`: latitude, longitude, year (2015-2100), and scenario (`ssp126`, `ssp245`, `ssp370`, or `ssp585`); returns projected annual mean temperature.
- `diy_greenhouse`: latitude, longitude, year, scenario, `delta_CO2` (percent), and `delta_CH4` (percent); returns local temperature under an emissions intervention.
- `diy_aerosol`: latitude, longitude, year, scenario, `delta_SO2` (percent), `delta_BC` (percent), and `modify_points` (longitude/latitude pairs); returns local intervention temperature.
- `diff_diy_aerosol_mean`: year, scenario, aerosol deltas, and modification points; returns the global mean temperature difference from baseline.

Named locations are first resolved with `query_lat_and_lon`, then passed to the relevant temperature function. The emulator can verify numerical temperatures, differences, intervention directions, and comparisons represented by these outputs. It cannot directly verify policy preferences, social outcomes, implementation feasibility, or unrelated climate quantities.
