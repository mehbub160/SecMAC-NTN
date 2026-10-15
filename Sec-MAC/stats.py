"""Mean and 95 % Student-t confidence half-width of independent replications."""
import math

from scipy import stats as _st


def mean_ci(values, conf=0.95):
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    n = len(v)
    if n == 0:
        return float('nan'), float('nan'), 0
    m = sum(v) / n
    if n == 1:
        return m, float('nan'), 1
    s2 = sum((x - m) ** 2 for x in v) / (n - 1)
    h = _st.t.ppf(0.5 + conf / 2, n - 1) * math.sqrt(s2 / n)
    return m, h, n
