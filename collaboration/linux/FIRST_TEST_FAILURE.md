# First test run, retained negative evidence

<!-- 2026-09-11 21:17 CST (linux): Record terminal session 15850, exit 1,
six test methods, one harness assertion failure and one upstream API error.
Exact original test bytes are retained in pilot_sources/test_prepared_tdi.py. -->

1. `test_native_random_pulse_zero` called stock `get_tdi_delays(y_gw=y)`.
   Installed response.py:597 tries `len(self.link_space_craft_0_in)` but that
   attribute is never defined on this 1.1.17 object. It raises AttributeError.
   This is a reproducible dependency public-input API defect, not sparse math.
   Do not modify installed files here. The revised native oracle writes the
   documented internal link buffer and uses the same no-argument native path
   as normal FEW response. A focused independent API experiment remains useful.
2. `test_scalar_stencil` incorrectly required exact mathematical identity at an
   integer node for every order. Stock A_in[0] is itself floating-point-built;
   one result was -0.5027322172618299 vs -0.5027322172618298. Revised test uses
   the actual stock A_in[0] factor; the independent longdouble sum gate is
   unchanged. This is a test expectation error, not evidence of sparse error.

All four other test methods passed. No previous result is deleted or described
as having passed this original suite. Original pilot file header times 21:14
were entered ahead of the observed wall clock; formal records use observed
CST and producer hashes to disambiguate ordering.
