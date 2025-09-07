import sys
from pathlib import Path

# Ensure project root is on path when tests are executed directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from comparison import difference, mean_bias, mean_absolute_error, evaluate


def test_difference_and_metrics():
    model = [1.0, 2.0, 3.0]
    analysis = [0.5, 1.5, 2.5]
    diff = difference(model, analysis)
    assert diff == [0.5, 0.5, 0.5]
    assert abs(mean_bias(diff) - 0.5) < 1e-6
    assert abs(mean_absolute_error(diff) - 0.5) < 1e-6
    diff2, metrics = evaluate(model, analysis)
    assert diff2 == diff
    assert abs(metrics["bias"] - 0.5) < 1e-6
    assert abs(metrics["mae"] - 0.5) < 1e-6
