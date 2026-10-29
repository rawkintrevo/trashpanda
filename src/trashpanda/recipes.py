"""Named recipe constants for callers that prefer module-level access."""
from .config import get_recipe

TINY = get_recipe("tiny")
PRODUCTION_300M_A100 = get_recipe("300m-a100")
PRODUCTION = PRODUCTION_300M_A100
TINY_RECIPE = TINY
PRODUCTION_RECIPE = PRODUCTION_300M_A100
RECIPES = {"tiny": TINY, "300m-a100": PRODUCTION_300M_A100}

__all__ = [
    "PRODUCTION",
    "PRODUCTION_300M_A100",
    "PRODUCTION_RECIPE",
    "RECIPES",
    "TINY",
    "TINY_RECIPE",
]
