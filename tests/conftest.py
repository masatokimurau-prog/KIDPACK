import matplotlib

# Tests must never depend on (or open windows on) a display.
matplotlib.use('Agg', force=True)

import matplotlib.pyplot as plt  # noqa: E402  (after the backend is fixed)
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def shown(monkeypatch):
    """Replace plt.show so that no test ever opens (or blocks on) a window.

    Each call is recorded as the list of (width, height) in inches of the open figures.
    """
    calls = []
    monkeypatch.setattr(
        plt, 'show', lambda *a, **k: calls.append(
            [tuple(plt.figure(n).get_size_inches()) for n in plt.get_fignums()]))
    yield calls
    plt.close('all')
