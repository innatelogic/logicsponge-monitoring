import operator
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import datasponge.core as ds

Time = datetime
TimeDelta = timedelta


def restrict_keys(original_dict: dict, allowed_keys: set[str]) -> dict:
    """Restricts a dictionary to only include keys that are in the allowed_keys set."""
    return {key: value for key, value in original_dict.items() if key in allowed_keys}


def generate_name(default_name: str | None, suffix: str) -> str:
    return default_name + suffix if default_name else suffix


# Usage
class TimeInterval:
    def __init__(self, start: TimeDelta, end: TimeDelta, *, start_strict: bool = False, end_strict: bool = False):
        self.start = start
        self.end = end
        self.start_strict = start_strict
        self.end_strict = end_strict

    def __repr__(self) -> str:
        start_border = "(" if self.start_strict else "["
        end_border = ")" if self.end_strict else "]"
        return f"{start_border}{self.start}, {self.end}{end_border}"

    def is_contained(self, value: TimeDelta) -> bool:
        if self.start_strict:
            if self.end_strict:
                return self.start < value < self.end
            return self.start < value <= self.end
        if self.end_strict:
            return self.start <= value < self.end
        return self.start <= value <= self.end

    def is_right_of(self, value: TimeDelta) -> bool:
        """value is on the right of interval (I < value)"""
        if self.end_strict:
            return self.end <= value
        return self.end < value


class LeftClosed(TimeInterval):
    def __init__(self, start: TimeDelta, end: TimeDelta):
        super().__init__(start, end, start_strict=False, end_strict=True)


class RightClosed(TimeInterval):
    def __init__(self, start: TimeDelta, end: TimeDelta):
        super().__init__(start, end, start_strict=True, end_strict=False)


class BothClosed(TimeInterval):
    def __init__(self, start: TimeDelta, end: TimeDelta):
        super().__init__(start, end, start_strict=False, end_strict=False)


class BothOpen(TimeInterval):
    def __init__(self, start: TimeDelta, end: TimeDelta):
        super().__init__(start, end, start_strict=True, end_strict=True)


class BooleanAggregate(ds.FunctionTerm):
    boolean_operation: Callable[[bool, bool], bool]

    def __init__(self, *args, op: Callable[[bool, bool], bool], **kwargs):
        super().__init__(*args, **kwargs)
        self.boolean_operation = op

    def run(self, ds_views: tuple[ds.DataStreamView]):
        if len(ds_views) <= 1:
            msg = "Expecting two data streams"
            raise ValueError(msg)

        ds_view1, ds_view2 = ds_views[0], ds_views[1]
        self.next(ds_view1)
        self.next(ds_view2)

        sat1 = ds_view1[-1]["Sat"]
        sat2 = ds_view2[-1]["Sat"]

        if self.boolean_operation is None:
            msg = "Logical operation not defined"
            raise NotImplementedError(msg)

        sat = self.boolean_operation(sat1, sat2)
        out = ds.DataItem({"Sat": sat})
        self.output(out)


class PMTL(ABC):
    def __or__(self, other):
        return Or(self, other)

    def __and__(self, other):
        return And(self, other)

    def __invert__(self):
        return Not(self)

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def to_term(self, name: str | None = None) -> ds.Term:
        """
        Passing a name ensures that each sub-term has a unique identifier
        within the term associated with the given formula.
        The given formula starts with 'Root', and subsequent names are
        strings composed of {0, 1}. These strings represent nodes in the
        corresponding tree structure.
        """


class Proposition(PMTL):
    """Class representing atomic propositions."""

    condition: Callable[[ds.DataItem], bool]

    def __init__(self, condition: Callable[[ds.DataItem], bool] | None = None):
        super().__init__()
        if condition is None:
            msg = "A condition must be provided."
            raise ValueError(msg)
        self.condition = condition

    def __str__(self):
        return "Proposition"

    def to_term(self, name: str | None = None) -> ds.FunctionTerm:
        proposition_instance = self

        class Check(ds.FunctionTerm):
            def f(self, item: ds.DataItem) -> ds.DataItem:
                sat = proposition_instance.condition(item)
                if "Time" in item:
                    return ds.DataItem({"Time": item["Time"], "Sat": sat})
                return ds.DataItem({"Sat": sat})

        return Check(name if name else "Root")


class TrueFormula(PMTL):
    """Class representing formula true."""

    def __str__(self):
        return "True"

    def to_term(self, name: str | None = None) -> ds.FunctionTerm:
        class OutputTrue(ds.FunctionTerm):
            def f(self, item: ds.DataItem) -> ds.DataItem:
                if "Time" in item:
                    return ds.DataItem({"Time": item["Time"], "Sat": True})
                return ds.DataItem({"Sat": True})

        return OutputTrue(name if name else "Root")


class Not(PMTL):
    """Class representing the negation of a formula."""

    formula: PMTL

    def __init__(self, formula: PMTL):
        super().__init__()
        self.formula = formula

    def __str__(self):
        return f"¬({self.formula})"

    def to_term(self, name: str | None = None) -> ds.SequentialTerm:
        term = self.formula.to_term(generate_name(name, "0"))

        class Inverter(ds.FunctionTerm):
            def f(self, item: ds.DataItem) -> ds.DataItem:
                if "Time" in item:
                    return ds.DataItem({"Time": item["Time"], "Sat": not item["Sat"]})
                return ds.DataItem({"Sat": not item["Sat"]})

        inverter_name = generate_name(name, "1")
        inverter = Inverter(inverter_name)
        new_term = term * inverter
        new_term.name = generate_name(name, "Root")
        return new_term


class BinaryOperation(PMTL):
    formula1: PMTL
    formula2: PMTL
    operator: Callable

    def __init__(self, formula1: PMTL, formula2: PMTL, op: Callable):
        super().__init__()
        self.formula1 = formula1
        self.formula2 = formula2
        self.operator = op

    def __str__(self):
        op_symbol = "|" if self.operator == operator.or_ else "&"
        return f"({self.formula1}) {op_symbol} ({self.formula2})"

    def to_term(self, name: str | None = None) -> ds.SequentialTerm:
        term1 = self.formula1.to_term(generate_name(name, "00"))
        term2 = self.formula2.to_term(generate_name(name, "01"))

        parallel = term1 | term2
        parallel.name = generate_name(name, "0")
        aggregate_name = generate_name(name, "1")
        aggregate = BooleanAggregate(aggregate_name, op=self.operator)
        new_term = parallel * aggregate
        new_term.name = generate_name(name, "Root")
        return new_term


class Or(BinaryOperation):
    def __init__(self, formula1: PMTL, formula2: PMTL):
        super().__init__(formula1, formula2, operator.or_)


class And(BinaryOperation):
    def __init__(self, formula1: PMTL, formula2: PMTL):
        super().__init__(formula1, formula2, operator.and_)


class Previous(PMTL):
    """Class to represent the Previous operator (checking the previous position - sometimes denoted Y or X^{-1})."""

    formula: PMTL
    interval: TimeInterval | None

    def __init__(self, formula: PMTL, interval: TimeInterval | None = None):
        super().__init__()
        self.formula = formula
        self.interval = interval

    def __str__(self):
        if self.interval:
            return f"Previous({self.formula}, {self.interval})"
        return f"Previous({self.formula})"

    def to_term(self, name: str | None = None) -> ds.SequentialTerm:
        term = self.formula.to_term(generate_name(name, "0"))
        interval = self.interval

        class CheckPrevious(ds.FunctionTerm):
            state: dict[str, Any]

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.state = {"Time": None, "Sat": False}  # maintaining Time satisfaction at previous position

            def f(self, item: ds.DataItem) -> ds.DataItem:
                prev_sat = self.state["Sat"]
                if interval is None:
                    out = {"Sat": prev_sat} if "Time" not in item else {"Time": item["Time"], "Sat": prev_sat}
                    self.state = restrict_keys(item, {"Time", "Sat"})
                elif "Time" not in item:
                    msg = "No timing information available in current data item."
                    raise RuntimeError(msg)
                else:
                    if self.state["Time"]:
                        timing_condition = interval.is_contained(item["Time"] - self.state["Time"])
                    else:
                        timing_condition = False
                    out = {"Time": item["Time"], "Sat": timing_condition & prev_sat}
                    self.state = restrict_keys(item, {"Time", "Sat"})

                return ds.DataItem(out)

        check_name = generate_name(name, "1")
        check = CheckPrevious(check_name)
        new_term = term * check
        new_term.name = name if name else "Root"
        return new_term


class Since(PMTL):
    """Class to represent the Since operator."""

    formula1: PMTL
    formula2: PMTL
    interval: TimeInterval | None

    def __init__(self, formula1: PMTL, formula2: PMTL, interval: TimeInterval | None = None):
        super().__init__()
        self.formula1 = formula1
        self.formula2 = formula2
        self.interval = interval

    def __str__(self):
        return f"({self.formula1}) Since ({self.formula2})"

    def to_term(self, name: str | None = None) -> ds.SequentialTerm:
        term1 = self.formula1.to_term(generate_name(name, "00"))
        term2 = self.formula2.to_term(generate_name(name, "01"))
        interval = self.interval

        class CheckSince(ds.FunctionTerm):
            state: dict[str, Any]

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.state = {"Times": deque(), "Sat": False}  # maintaining satisfaction of Since formula

            @staticmethod
            def _truncate_times(times: deque[Time], current_time: Time):
                """for times being sorted in increasing order, this function
                returns max{t in times | current_time - t in interval} union {t in times | current_time - t < I}
                E.g.: if interval=[2, 5], ([2, 4, 6, 7, 9, 10], 10) is mapped to [7, 9, 10]"""

                if interval is None:
                    raise RuntimeError
                while times:
                    if interval.is_right_of(current_time - times[0]):
                        times.popleft()
                        continue
                    if (
                        interval.is_contained(current_time - times[0])
                        and len(times) > 1
                        and interval.is_contained(current_time - times[1])
                    ):
                        times.popleft()
                        continue
                    break

            def run(self, ds_views: tuple[ds.DataStreamView]):
                if len(ds_views) <= 1:
                    msg = "Expecting two data streams"
                    raise ValueError(msg)

                ds_view1, ds_view2 = ds_views[0], ds_views[1]
                self.next(ds_view1)
                self.next(ds_view2)

                data_item1 = ds_view1[-1]
                data_item2 = ds_view2[-1]

                sat1 = data_item1["Sat"]
                sat2 = data_item2["Sat"]

                if interval is None:
                    sat = sat2 or (sat1 and self.state["Sat"])
                    if "Time" not in data_item1:
                        out = {"Sat": sat}
                    else:
                        current_time = data_item1["Time"]
                        out = {"Time": current_time, "Sat": sat}

                    self.state = {"Times": self.state["Times"], "Sat": sat}
                elif "Time" not in data_item1:
                    msg = "No timing information available in current data item."
                    raise RuntimeError(msg)
                else:
                    current_time = data_item1["Time"]
                    times = self.state["Times"]
                    if not sat1:
                        times = deque()  # Reset times to an empty deque
                    if sat2:
                        times.append(current_time)

                    self._truncate_times(times, current_time)

                    # check satisfaction at current position
                    if times:
                        sat = interval.is_contained(current_time - times[0])
                        out = {"Time": current_time, "Sat": sat}
                    else:
                        out = {"Time": current_time, "Sat": False}

                self.output(ds.DataItem(out))

        parallel = term1 | term2
        parallel.name = generate_name(name, "0")
        check_name = generate_name(name, "1")
        check = CheckSince(check_name)
        new_term = parallel * check
        new_term.name = generate_name(name, "Root")
        return new_term


class Earlier(PMTL):
    """Class to represent some time in the past (including current position)."""

    formula: PMTL
    interval: TimeInterval | None

    def __init__(self, formula: PMTL, interval: TimeInterval | None = None):
        super().__init__()
        self.formula = formula
        self.interval = interval

    def __str__(self):
        return f"Earlier({self.formula})"

    def to_term(self, name: str | None = None) -> ds.SequentialTerm:
        formula = Since(TrueFormula(), self.formula, self.interval)
        return formula.to_term(name)


class Hist(PMTL):
    """Class to represent always in the past."""

    formula: PMTL
    interval: TimeInterval | None

    def __init__(self, formula: PMTL, interval: TimeInterval | None = None):
        super().__init__()
        self.formula = formula
        self.interval = interval

    def __str__(self):
        return f"Hist({self.formula})"

    def to_term(self, name: str | None = None) -> ds.SequentialTerm:
        formula = ~Earlier(~self.formula, self.interval)
        return formula.to_term(name)
