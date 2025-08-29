import operator
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any

import logicsponge.core as ls


Time = datetime
TimeDelta = timedelta


def _get_time_from(item: ls.DataItem) -> Any | None:
    return item.get("Time") if isinstance(item, dict) and "Time" in item else None


def restrict_keys(original_dict: dict, allowed_keys: set[str]) -> dict:
    """Restricts a dictionary to only include keys that are in the allowed_keys set."""
    return {key: value for key, value in original_dict.items() if key in allowed_keys}


def generate_name(default_name: str | None, suffix: str) -> str:
    return default_name + suffix if default_name else suffix


@dataclass(frozen=True)
class Reason:
    kind: str                 # e.g. "proposition" (later: "and", "or", "not", etc.)
    value: bool               # the boolean outcome
    message: str              # human-readable explanation
    time: Any | None = None   # optional timestamp; usually item.get("Time")
    label: str = ""           # optional short name (e.g., "HR>120")
    # Later add: evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {"value": self.value, "message": self.message}
        if self.time is not None:
            d["Time"] = self.time
        if self.label:
            d["label"] = self.label
        d["kind"] = self.kind
        return d

    def __str__(self) -> str:
        head = f"[{self.label}] " if self.label else ""
        return f"{head}{self.message} (value={self.value})"


# ---- Condition: minimal API per your spec, now returning Reason ----
class Condition:
    """
    A simple condition with evaluation and explanations.
    - func: item -> bool
    - true_message: item -> str
    - false_message: item -> str
    """

    def __init__(
        self,
        func: Callable[[ls.DataItem], bool],
        true_message: Callable[[ls.DataItem], str],
        false_message: Callable[[ls.DataItem], str],
        *,
        label: str = "",                  # optional human-readable name
        include_time: bool = True,        # pick up item["Time"] if present
        kind: str = "proposition",        # classify the reason
    ):
        self.func = func
        self.true_message = true_message
        self.false_message = false_message
        self.label = label
        self.include_time = include_time
        self.kind = kind

    # Back-compat: allow using Condition as a plain predicate
    def __call__(self, item: ls.DataItem) -> bool:
        return bool(self.func(item))

    # Preferred API: return a Reason
    def evaluate(self, item: ls.DataItem) -> Reason:
        value = bool(self.func(item))
        msg = self.true_message(item) if value else self.false_message(item)
        t = item.get("Time") if (self.include_time and isinstance(item, dict)) else None
        return Reason(kind=self.kind, value=value, message=msg, time=t, label=self.label)

    # Back-compat helper: old shape {"value","message",["Time"]}
    def evaluate_simple(self, item: ls.DataItem) -> dict[str, Any]:
        r = self.evaluate(item)
        return r.to_dict()  # already returns {"value","message", ...}


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


class BooleanAggregate(ls.FunctionTerm):
    boolean_operation: Callable[[bool, bool], bool]

    def __init__(self, *args, op: Callable[[bool, bool], bool], **kwargs):
        super().__init__(*args, **kwargs)
        self.boolean_operation = op

    def run(self, ds_views: tuple[ls.DataStreamView]):
        while True:
            if len(ds_views) <= 1:
                msg = "Expecting two data streams"
                raise ValueError(msg)

            ds_view1, ds_view2 = ds_views[0], ds_views[1]
            self.next(ds_view1)
            self.next(ds_view2)

            di1 = ds_view1[-1]
            di2 = ds_view2[-1]
            sat1 = bool(di1["Sat"])
            sat2 = bool(di2["Sat"])

            if self.boolean_operation is None:
                msg = "Logical operation not defined"
                raise NotImplementedError(msg)

            sat = self.boolean_operation(sat1, sat2)

            # Determine kind from the operator (fallback: "binary")
            if self.boolean_operation is operator.or_:
                kind = "or"
            elif self.boolean_operation is operator.and_:
                kind = "and"
            else:
                kind = "binary"

            # Prefer Time from the left stream; fall back to right if absent
            t = di1.get("Time") if isinstance(di1, dict) and "Time" in di1 else (
                di2.get("Time") if isinstance(di2, dict) and "Time" in di2 else None
            )

            # Build a minimal Reason (we can compose child reasons later if desired)
            r = Reason(kind=kind, value=sat, message=f"Evaluation result: {sat}", time=t)

            out = {"Sat": sat, "reason": r}
            if t is not None:
                out["Time"] = t

            self.output(ls.DataItem(out))


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
    def to_term(self, name: str | None = None) -> ls.Term:
        """
        Passing a name ensures that each sub-term has a unique identifier
        within the term associated with the given formula.
        The given formula starts with 'Root', and subsequent names are
        strings composed of {0, 1}. These strings represent nodes in the
        corresponding tree structure.
        """

# Helper to turn fixed strings into callables
def _msg_callable(msg_or_fn: str | Callable[[ls.DataItem], str], default: str) -> Callable[[ls.DataItem], str]:
    if callable(msg_or_fn):
        return msg_or_fn
    if isinstance(msg_or_fn, str):
        return lambda _di: msg_or_fn
    return lambda _di: default

class Proposition(PMTL):
    """
    Proposition that accepts either:
      - a Condition (preferred new style), or
      - a plain predicate (legacy), which is auto-wrapped.
    """

    def __init__(
        self,
        condition: Condition | Callable[[ls.DataItem], bool],
        *,
        label: str | None = None,
        true_message: str | Callable[[ls.DataItem], str] | None = None,
        false_message: str | Callable[[ls.DataItem], str] | None = None,
    ):
        super().__init__()

        if isinstance(condition, Condition):
            self.condition = condition
            self._label = condition.label or "Proposition"
        else:
            lbl = label or "Proposition"
            tm = _msg_callable(true_message, f"{lbl} is True")
            fm = _msg_callable(false_message, f"{lbl} is False")
            self.condition = Condition(
                func=condition,
                true_message=tm,
                false_message=fm,
                label=lbl,
                kind="proposition",
            )
            self._label = lbl

    def __str__(self):
        return "Proposition"

    def to_term(self, name: str | None = None) -> ls.FunctionTerm:
        prop = self

        class Check(ls.FunctionTerm):
            def f(self, item: ls.DataItem) -> ls.DataItem:
                reason = prop.condition.evaluate(item)  # Reason
                sat = bool(reason.value)
                out = {"Sat": sat, "reason": reason}
                if "Time" in item:
                    out["Time"] = item["Time"]
                return ls.DataItem(out)

        return Check(name if name else "Root")



class TrueFormula(PMTL):
    """Class representing formula true."""

    def __str__(self):
        return "True"

    def to_term(self, name: str | None = None) -> ls.FunctionTerm:
        class OutputTrue(ls.FunctionTerm):
            def f(self, item: ls.DataItem) -> ls.DataItem:
                r = Reason(
                    kind="true",
                    value=True,
                    message="Formula 'True' always holds",
                    time=_get_time_from(item),
                    label="True",
                )
                out = {"Sat": True, "reason": r}
                if "Time" in item:
                    out["Time"] = item["Time"]
                return ls.DataItem(out)

        return OutputTrue(name if name else "Root")


class Not(PMTL):
    """Class representing the negation of a formula."""

    formula: PMTL

    def __init__(self, formula: PMTL):
        super().__init__()
        self.formula = formula

    def __str__(self):
        return f"Â¬({self.formula})"

    def to_term(self, name: str | None = None) -> ls.SequentialTerm:
        term = self.formula.to_term(generate_name(name, "0"))

        class Inverter(ls.FunctionTerm):
            def f(self, item: ls.DataItem) -> ls.DataItem:
                prev = bool(item["Sat"])
                sat = not prev
                t = _get_time_from(item)
                r = Reason(kind="not", value=sat, message=f"not {prev} ⇒ {sat}", time=t, label="¬")
                out = {"Sat": sat, "reason": r}
                if t is not None:
                    out["Time"] = t
                return ls.DataItem(out)

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

    def to_term(self, name: str | None = None) -> ls.SequentialTerm:
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

    def to_term(self, name: str | None = None) -> ls.SequentialTerm:
        term = self.formula.to_term(generate_name(name, "0"))
        interval = self.interval

        class CheckPrevious(ls.FunctionTerm):
            state: dict[str, Any]

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.state = {"Time": None, "Sat": False}  # maintaining Time satisfaction at previous position

            def f(self, item: ls.DataItem) -> ls.DataItem:
                prev_sat = self.state["Sat"]
                if interval is None:
                    out = (
                        {"Sat": prev_sat, "reason": Reason(kind="previous", value=prev_sat, message=f"Evaluation result: {prev_sat}")}
                        if "Time" not in item
                        else {"Time": item["Time"], "Sat": prev_sat, "reason": Reason(kind="previous", value=prev_sat, message=f"Evaluation result: {prev_sat}")}
                    )
                    self.state = restrict_keys(item, {"Time", "Sat"})
                elif "Time" not in item:
                    msg = "No timing information available in current data item."
                    raise RuntimeError(msg)
                else:
                    if self.state["Time"]:
                        timing_condition = interval.is_contained(item["Time"] - self.state["Time"])
                    else:
                        timing_condition = False
                    sat = timing_condition & prev_sat
                    out = {
                        "Time": item["Time"],
                        "Sat": sat,
                        "reason": Reason(kind="previous", value=sat, message=f"Evaluation result: {sat}"),
                    }
                    self.state = restrict_keys(item, {"Time", "Sat"})

                return ls.DataItem(out)

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

    def to_term(self, name: str | None = None) -> ls.SequentialTerm:
        term1 = self.formula1.to_term(generate_name(name, "00"))
        term2 = self.formula2.to_term(generate_name(name, "01"))
        interval = self.interval

        class CheckSince(ls.FunctionTerm):
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

            def run(self, ds_views: tuple[ls.DataStreamView]):
                while True:
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
                            out = {"Sat": sat, "reason": Reason(kind="since", value=sat, message=f"Evaluation result: {sat}")}
                        else:
                            current_time = data_item1["Time"]
                            out = {
                                "Time": current_time,
                                "Sat": sat,
                                "reason": Reason(kind="since", value=sat, message=f"Evaluation result: {sat}"),
                            }

                        self.state = {"Times": self.state["Times"], "Sat": sat}
                    elif "Time" not in data_item1:
                        msg = "No timing information available in current data item."
                        raise RuntimeError(msg)
                    else:
                        current_time = data_item1["Time"]
                        if not sat1:
                            self.state["Times"] = deque()  # Reset times to an empty deque
                        if sat2:
                            self.state["Times"].append(current_time)

                        self._truncate_times(self.state["Times"], current_time)

                        # check satisfaction at current position
                        if self.state["Times"]:
                            sat = interval.is_contained(current_time - self.state["Times"][0])
                            out = {
                                "Time": current_time,
                                "Sat": sat,
                                "reason": Reason(kind="since", value=sat, message=f"Evaluation result: {sat}"),
                            }
                        else:
                            sat = False
                            out = {
                                "Time": current_time,
                                "Sat": sat,
                                "reason": Reason(kind="since", value=sat, message=f"Evaluation result: {sat}"),
                            }

                    self.output(ls.DataItem(out))

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

    def to_term(self, name: str | None = None) -> ls.SequentialTerm:
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

    def to_term(self, name: str | None = None) -> ls.SequentialTerm:
        formula = ~Earlier(~self.formula, self.interval)
        return formula.to_term(name)
