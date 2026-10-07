"""Reusable interval joins for source-dated events."""

from datetime import datetime


def seconds(value):
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


def matches(first_start, first_end, second_start, second_end, operator, window=0):
    """Whether first event bears operator to second, including endpoints.

    `before`/`after` use nonoverlapping intervals and optional maximum gaps;
    `followed_by` is the inverse of `after` in natural event ordering.
    A negative window means no upper gap bound.
    """
    a, b, c, d = map(seconds, (first_start, first_end, second_start, second_end))
    if a > b or c > d or window < -1:
        raise ValueError("Invalid interval or window")
    if operator in ("before", "followed_by"):
        gap = (c - b).total_seconds()
        return gap >= 0 and (window < 0 or gap <= window)
    if operator == "after":
        gap = (a - d).total_seconds()
        return gap >= 0 and (window < 0 or gap <= window)
    if operator == "during":
        return c <= a and b <= d
    if operator == "overlaps":
        return max(a, c) <= min(b, d)
    if operator in ("near", "within"):
        gap = max(0, (c - b).total_seconds(), (a - d).total_seconds())
        return gap <= window
    raise ValueError(f"Unknown temporal operator: {operator}")
