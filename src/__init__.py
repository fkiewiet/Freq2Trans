"""
Top-level package for the project.

We keep three sibling subpackages:
- core: problem definition + grids + media + RHS
- operators: discretization/assembly + solvers
- algorithm: multilevel/iterative refinement + transfer operators
"""

__all__ = ["core", "operators", "algorithm"]
