# Compound-growth simulator

The simulator exposes one function, `compound_growth`.

Required arguments:

- `initial_value`: number greater than or equal to 0.
- `rate_percent`: annual percentage rate from -100 through 1000.
- `years`: integer from 0 through 100.

It returns `final_value`, `absolute_change`, and `percent_change`, using annual compounding.

