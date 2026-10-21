"""cenmodel: replicate and forecast the Polish imbalance price (CEN).

Run `python -m cenmodel` for the list of commands.
"""
__version__ = "1.1.0"


class UserError(Exception):
    """A problem the user can fix (config, data, network). Printed without traceback."""
