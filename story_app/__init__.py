"""Kid drawing to bedtime story app."""

__all__ = ["build_demo"]


def build_demo():
    from .app import build_demo as _build_demo

    return _build_demo()
