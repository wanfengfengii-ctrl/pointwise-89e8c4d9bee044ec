"""Named-node differences interpreted against one current calibrated state."""
from fractions import Fraction
from ledger import name


class Analyzer:
    def __init__(self, calibrator):
        self.calibrator = calibrator

    def query(self, a, b):
        name(a)
        name(b)
        state = self.calibrator.current()
        if a not in state.membership or b not in state.membership:
            raise ValueError("unknown node")
        if a == b:
            return Fraction(0), Fraction(0)
        component = state.membership[a]
        if component != state.membership[b]:
            return None
        fit = state.fits[component]
        i, j = fit.positions[a], fit.positions[b]
        estimate = fit.values[j] - fit.values[i]
        cov = fit.covariance
        variance = cov[i][i] + cov[j][j] - cov[i][j] - cov[j][i]
        return estimate, variance
