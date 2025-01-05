try:
    from aspeed import Aspeed

    ASPEED_AVAILABLE = True
except ImportError:
    ASPEED_AVAILABLE = False

from presolver import AbstractPresolver


__all__ = ["ASPEED_AVAILABLE", "Aspeed", "AbstractPresolver"]
