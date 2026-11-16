"""Named-node differences interpreted against one current calibrated state."""
from fractions import Fraction
from ledger import (name, BOUNDED_MAX_NODES, BOUNDED_MAX_OBSERVATIONS,
                    BOUNDED_MAX_ENDPOINTS)
from bounded import lp_extremum


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

    def range(self, a, b):
        """Exact min/max of x_b - x_a over the whole bounded optimal set.

        Returns a two-element tuple of Fraction endpoints; a None lower
        endpoint means minus infinity and a None upper endpoint plus
        infinity. This is a projection over all global optima, not a
        confidence interval.
        """
        name(a)
        name(b)
        ledger = self.calibrator.ledger
        if a not in ledger.nodes or b not in ledger.nodes:
            raise ValueError("unknown node")
        endpoints = sum((bound.lower is not None) + (bound.upper is not None)
                        for bound in ledger.bounds.values())
        if (len(ledger.nodes) > BOUNDED_MAX_NODES
                or len(ledger.snapshot().rows) > BOUNDED_MAX_OBSERVATIONS
                or endpoints > BOUNDED_MAX_ENDPOINTS):
            raise ValueError("bounded analysis size limits exceeded")
        if a == b:
            return Fraction(0), Fraction(0)
        optimum = self.calibrator.optimum()
        n_vars = len(optimum.nodes)
        index = {node: i for i, node in enumerate(optimum.nodes)}
        objective = [Fraction(0)] * n_vars
        objective[index[a]] = Fraction(-1)
        objective[index[b]] = Fraction(1)
        low = lp_extremum(n_vars, optimum.equalities, optimum.inequalities,
                          objective, maximize=False)
        high = lp_extremum(n_vars, optimum.equalities, optimum.inequalities,
                           objective, maximize=True)
        return low, high
