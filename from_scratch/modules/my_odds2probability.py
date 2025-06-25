"""summary"""

import numpy as np
from numba import njit


@njit
def prob2odds(prob: float) -> float:
    """convert probability to odds

    Args:
        prob (float): probability

    Returns:
        float: odds
    """
    return prob / (1 - prob)


@njit
def odds2prob(odds: float) -> float:
    """convert odds to probability

    Args:
        odds (float): odds

    Returns:
        float: probability
    """
    return odds / (1 + odds)


def main() -> None:
    """_summary_"""
    prob = 3 / 19
    odds = 3 / 16

    print(f"\nTrue prob: {prob:.5f}")
    print(f"True odds: {odds:.5f}\n")

    print(f"Prob via {odds2prob(odds) = :.5f}")
    print(f"Odds via {prob2odds(prob) = :.5f}")


if __name__ == "__main__":
    main()
