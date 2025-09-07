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


def _get_id_from(item: ls.DataItem, id_key: str = "ITEM_ID") -> Any | None:
    return item.get(id_key) if isinstance(item, dict) and id_key in item else None


def _safe_get(obj, key, default=None):
    # Works for dict-like and objects exposing .get
    try:
        if hasattr(obj, "get"):
            return obj.get(key, default)
    except TypeError:
        pass
    return default


def _item_ref(item: ls.DataItem) -> dict[str, Any] | None:
    # Returns {"item_id": ..., "time": ...} or None
    _id = _safe_get(item, "ITEM_ID", None)
    _t  = _safe_get(item, "Time", None)
    if _id is None and _t is None:
        return None
    return {"item_id": _id, "time": _t}



def _short(msg: str, limit: int = 120) -> str:
    return msg if len(msg) <= limit else msg[:limit - 1].rstrip() + "…"


def _get_time_from(item: ls.DataItem) -> Any | None:
    return item.get("Time") if isinstance(item, dict) and "Time" in item else None


def restrict_keys(original_dict: dict, allowed_keys: set[str]) -> dict:
    """Restricts a dictionary to only include keys that are in the allowed_keys set."""
    return {key: value for key, value in original_dict.items() if key in allowed_keys}


def generate_name(default_name: str | None, suffix: str) -> str:
    return default_name + suffix if default_name else suffix


@dataclass(frozen=True)
class Reason:
    kind: str                  # e.g. "proposition", "and", "or", "not", "since"
    value: bool                # Boolean outcome
    message: str               # Single-line human-readable summary
    time: Any | None = None    # Usually item["Time"]
    label: str = ""            # Optional short name, e.g. "HR>120"

    # Structure and evidence
    children: tuple["Reason", ...] = ()           # sub-reasons (kept small)
    item: dict[str, Any] | None = None            # {"id":..., "time":...} of this decision point
    evidence: tuple[dict[str, Any], ...] = ()     # concrete witness items for timeline
    span: dict[str, dict[str, Any]] | None = None # {"start": {...}, "end": {...}} if contiguous witness interval
    example_reason: "Reason | None" = None        # one representative guard reason inside span (success case)

    # Convert to plain dict (for JSON export, UI, logs)
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kind": self.kind,
            "value": self.value,
            "message": self.message,
        }
        if self.time is not None:
            d["Time"] = self.time
        if self.label:
            d["label"] = self.label
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        if self.item is not None:
            d["item"] = self.item
        if self.evidence:
            d["evidence"] = list(self.evidence)
        if self.span is not None:
            d["span"] = self.span
        if self.example_reason is not None:
            d["example_reason"] = self.example_reason.to_dict()
        return d

    def __str__(self) -> str:
        head = f"[{self.label}] " if self.label else ""
        ex = f" | Example: {self.example_reason.message}" if self.example_reason else ""
        return f"{head}{self.message} (value={self.value}){ex}"



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

        # identity + evidence for timeline
        ir = {"item_id": item.get("ITEM_ID"), "time": t} if isinstance(item, dict) else None
        evidence = (ir,) if ir else ()

        return Reason(
            kind=self.kind,
            value=value,
            message=msg,
            time=t,
            label=self.label,
            item=ir,
            evidence=evidence,
        )

    # Back-compat helper: old shape {"value","message",["Time"]}
    def evaluate_simple(self, item: ls.DataItem) -> dict[str, Any]:
        r = self.evaluate(item)
        return r.to_dict()



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
                raise ValueError("Expecting two data streams")

            # Advance both inputs
            ds_view1, ds_view2 = ds_views[0], ds_views[1]
            self.next(ds_view1)
            self.next(ds_view2)

            di1 = ds_view1[-1]
            di2 = ds_view2[-1]

            sat1 = bool(di1["Sat"])
            sat2 = bool(di2["Sat"])
            r1: Reason | None = di1.get("reason")
            r2: Reason | None = di2.get("reason")

            if self.boolean_operation is None:
                raise NotImplementedError("Logical operation not defined")

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
            # Decision item for the timeline (use whichever has an item/Time)
            decision_item = _item_ref(di1) or _item_ref(di2)

            # --- Compose message & evidence policy ---
            if kind == "and":
                if sat:
                    msg = "Both conditions satisfied."
                    ev: tuple[dict, ...] = tuple((r1.evidence or ()) + (r2.evidence or ())) if (r1 or r2) else ()
                else:
                    # List only culprits (false children)
                    culprits: list[str] = []
                    if not sat1 and r1 is not None:
                        culprits.append(_short(r1.message))
                    if not sat2 and r2 is not None:
                        culprits.append(_short(r2.message))
                    msg = "Conjunction failed: " + "; ".join(culprits) if culprits else "Conjunction failed."
                    ev = ()
                    if not sat1 and r1 is not None:
                        ev += r1.evidence
                    if not sat2 and r2 is not None:
                        ev += r2.evidence
            elif kind == "or":
                if sat:
                    # Winner = first true child (leftmost)
                    winner = r1 if sat1 else r2
                    msg = "Holds via: " + _short(winner.message) if winner else "Disjunction holds."
                    ev = winner.evidence if winner else tuple()
                else:
                    msg = "Neither condition holds."
                    ev = tuple((r1.evidence or ()) + (r2.evidence or ())) if (r1 or r2) else ()
            else:
                # Generic binary (if ever used)
                msg = f"Evaluation result: {sat}"
                ev = tuple((r1.evidence or ()) + (r2.evidence or ())) if (r1 or r2) else ()

            reason = Reason(
                kind=kind,
                value=sat,
                message=msg,
                time=t,
                children=tuple(x for x in (r1, r2) if x is not None),
                item=decision_item,
                evidence=ev,
                span=None,
            )

            out = {"Sat": sat, "reason": reason}
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
            self._label = condition.label or (label or "Proposition")
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
                # Leaf reason from Condition (ideally already has message + item/evidence)
                base: Reason = prop.condition.evaluate(item)

                # Ensure label, item, evidence (robust to non-dict items)
                lbl = base.label or prop._label or ""
                ir  = base.item or _item_ref(item)            # {"item_id":..., "time":...} or None
                ev  = base.evidence or ((ir,) if ir else ())

                # Only rebuild Reason if we need to inject/fix fields
                reason = base if (lbl == base.label and ir == base.item and ev == base.evidence) else Reason(
                    kind=base.kind,
                    value=base.value,
                    message=base.message,
                    time=base.time if base.time is not None else _safe_get(item, "Time", None),
                    label=lbl,
                    children=base.children,
                    item=ir,
                    evidence=ev,
                    span=base.span,
                    example_reason=base.example_reason,
                )

                t = _safe_get(item, "Time", None)
                out = {"Sat": bool(reason.value), "reason": reason}
                if t is not None:
                    out["Time"] = t
                return ls.DataItem(out)

        return Check(name if name else "Root")



class TrueFormula(PMTL):
    """Class representing formula true."""

    def __str__(self):
        return "True"

    def to_term(self, name: str | None = None) -> ls.FunctionTerm:
        class OutputTrue(ls.FunctionTerm):
            def f(self, item: ls.DataItem) -> ls.DataItem:
                t = _safe_get(item, "Time", None)
                ir = _item_ref(item)

                r = Reason(
                    kind="true",
                    value=True,
                    message="Formula 'True' always holds",
                    time=t,
                    label="True",
                    children=(),
                    item=ir,
                    evidence=(),          # tautology → nothing to pin as evidence
                    span=None,
                    example_reason=None,
                )
                out = {"Sat": True, "reason": r}
                if t is not None:
                    out["Time"] = t
                return ls.DataItem(out)

        return OutputTrue(name if name else "Root")


class Not(PMTL):
    """Class representing the negation of a formula."""

    formula: PMTL

    def __init__(self, formula: PMTL):
        super().__init__()
        self.formula = formula

    def __str__(self):
        return f"¬({self.formula})"  # fix encoding of the negation symbol

    def to_term(self, name: str | None = None) -> ls.SequentialTerm:
        term = self.formula.to_term(generate_name(name, "0"))

        class Inverter(ls.FunctionTerm):
            def f(self, item: ls.DataItem) -> ls.DataItem:
                prev = bool(_safe_get(item, "Sat", False))
                child: Reason | None = _safe_get(item, "reason", None)

                sat = not prev
                t = _safe_get(item, "Time", None)
                ir = _item_ref(item)

                if child is not None:
                    msg = f"not ({_short(child.message)}) ⇒ {sat}"
                    evidence = child.evidence
                    kids = (child,)
                else:
                    msg = f"not {prev} ⇒ {sat}"
                    evidence = ()
                    kids = ()

                r = Reason(
                    kind="not",
                    value=sat,
                    message=msg,
                    time=t,
                    label="¬",
                    children=kids,
                    item=ir,
                    evidence=evidence,
                    span=None,
                    example_reason=None,
                )

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
                # Keep original fields + remember the child's Reason (for explanations)
                self.state = {"Time": None, "Sat": False, "Reason": None}

            def f(self, item: ls.DataItem) -> ls.DataItem:
                # --- original semantics preserved ---
                prev_sat = self.state["Sat"]

                # Robust accessors
                t_now = _safe_get(item, "Time", None)
                child_reason: Reason | None = _safe_get(item, "reason", None)
                decision_item = _item_ref(item)

                if interval is None:
                    # Message/evidence based on previous child reason (if any)
                    if self.state["Reason"] is not None:
                        msg = f"Previous: {_short(self.state['Reason'].message)}"
                        ev = self.state["Reason"].evidence
                        kids = (self.state["Reason"],)
                    else:
                        msg = "Previous: not available."
                        ev = ()
                        kids = ()

                    r = Reason(
                        kind="previous",
                        value=prev_sat,
                        message=msg,
                        time=t_now,
                        children=kids,
                        item=decision_item,
                        evidence=ev,
                        span=None,
                        example_reason=None,
                    )

                    out = {"Sat": prev_sat, "reason": r}
                    if t_now is not None:
                        out["Time"] = t_now

                    # --- original state update, augmented with Reason ---
                    self.state = restrict_keys(item, {"Time", "Sat"})
                    self.state["Reason"] = child_reason
                    return ls.DataItem(out)

                # --- interval case ---
                if "Time" not in item:
                    raise RuntimeError("No timing information available in current data item.")

                if self.state["Time"]:
                    timing_condition = interval.is_contained(item["Time"] - self.state["Time"])
                else:
                    timing_condition = False

                # Keep original boolean computation exactly
                sat = timing_condition & prev_sat

                # Build explanation
                if self.state["Reason"] is not None:
                    msg_prev = _short(self.state["Reason"].message)
                    msg = f"Previous (within {interval}): {msg_prev} ⇒ {sat}"
                    ev = self.state["Reason"].evidence
                    kids = (self.state["Reason"],)
                else:
                    msg = f"Previous (within {interval}): {prev_sat} ⇒ {sat}"
                    ev = ()
                    kids = ()

                r = Reason(
                    kind="previous",
                    value=sat,
                    message=msg,
                    time=item["Time"],
                    children=kids,
                    item=decision_item,
                    evidence=ev,
                    span=None,
                    example_reason=None,
                )

                out = {"Time": item["Time"], "Sat": sat, "reason": r}

                # --- original state update, augmented with Reason ---
                self.state = restrict_keys(item, {"Time", "Sat"})
                self.state["Reason"] = child_reason
                return ls.DataItem(out)

        check_name = generate_name(name, "1")
        check = CheckPrevious(check_name)
        new_term = term * check
        new_term.name = name if name else "Root"
        return new_term


class Since(PMTL):
    """Class to represent the Since operator."""

    formula: PMTL
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
                # Keep original fields for algorithm; add Witnesses only for explanations
                self.state = {
                    "Times": deque(),     # (used by your bounded-interval algorithm)
                    "Sat": False,         # last Sat (used by unbounded algorithm)
                    "Witnesses": deque()  # deque[(time, r2_reason)] for explanation only
                }

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
                        raise ValueError("Expecting two data streams")

                    ds_view1, ds_view2 = ds_views[0], ds_views[1]
                    self.next(ds_view1)
                    self.next(ds_view2)

                    data_item1 = ds_view1[-1]
                    data_item2 = ds_view2[-1]

                    sat1 = bool(data_item1["Sat"])
                    sat2 = bool(data_item2["Sat"])
                    r1: Reason = data_item1.get("reason")
                    r2: Reason = data_item2.get("reason")

                    # ---------- UNBOUNDED CASE ----------
                    if interval is None:
                        # keep your SAT computation
                        sat = sat2 or (sat1 and self.state["Sat"])

                        # explanation bits
                        t_now = _safe_get(data_item1, "Time", None)
                        now_ref = _item_ref(data_item1)

                        # maintain witnesses only for explanations (no effect on sat)
                        if sat2:
                            self.state["Witnesses"].append((t_now, r2))

                        # choose the current witness (rightmost so far)
                        witness = self.state["Witnesses"][-1] if self.state["Witnesses"] else None

                        if sat:
                            # Success: message uses φ2 witness message; span from witness to now (if both known)
                            if witness and witness[1] and witness[1].item:
                                msg = f"Since holds: {_short(witness[1].message)}; guard held to now."
                                ev = witness[1].evidence
                                span = {"start": witness[1].item, "end": now_ref} if now_ref else None
                            else:
                                msg = "Since holds."
                                ev = ()
                                span = None
                            # one φ1 example in the span: use current r1 if sat1 (it lies at 'now')
                            example = r1 if sat1 and isinstance(r1, Reason) else None
                        else:
                            # Failure (either no witness ever, or guard broken so earlier Sat dropped)
                            msg = "Since fails: no witness or guard broken."
                            ev = ()
                            span = None
                            example = None

                        out_reason = Reason(
                            kind="since",
                            value=sat,
                            message=msg,
                            time=t_now,
                            children=tuple(x for x in (r1, r2) if x is not None),
                            item=now_ref,
                            evidence=ev,
                            span=span,
                            example_reason=example,
                        )

                        self.state["Sat"] = sat  # keep your original state update
                        out = {"Sat": sat, "reason": out_reason}
                        if t_now is not None:
                            out["Time"] = t_now
                        self.output(ls.DataItem(out))
                        continue

                    # ---------- BOUNDED-INTERVAL CASE ----------
                    if "Time" not in data_item1:
                        raise RuntimeError("No timing information available in current data item.")
                    current_time = data_item1["Time"]
                    now_ref = _item_ref(data_item1)

                    # keep your original guard reset and witness append
                    if not sat1:
                        self.state["Times"] = deque()    # reset
                        self.state["Witnesses"] = deque() # explanations: reset aligned witnesses
                    if sat2:
                        self.state["Times"].append(current_time)
                        self.state["Witnesses"].append((current_time, r2))

                    # keep your truncation for Times; mirror for Witnesses (explanations)
                    self._truncate_times(self.state["Times"], current_time)
                    while self.state["Witnesses"] and interval.is_right_of(current_time - self.state["Witnesses"][0][0]):
                        self.state["Witnesses"].popleft()

                    # check satisfaction at current position (unchanged)
                    if self.state["Times"]:
                        t_star = self.state["Times"][0]
                        sat = interval.is_contained(current_time - t_star)

                        # find matching r2 for the witness time (for explanations)
                        r2_star = None
                        for (tw, rw) in self.state["Witnesses"]:
                            if tw == t_star:
                                r2_star = rw
                                break

                        if sat:
                            # success
                            if r2_star and r2_star.item:
                                msg = f"Since holds: {_short(r2_star.message)}; guard held to now."
                                ev = r2_star.evidence
                                span = {"start": r2_star.item, "end": now_ref} if now_ref else None
                            else:
                                msg = "Since holds."
                                ev = ()
                                span = {"start": {"id": None, "time": t_star}, "end": now_ref} if now_ref else None
                            example = r1 if sat1 and isinstance(r1, Reason) else None
                        else:
                            # failure: witness exists but interval condition not met yet
                            if r2_star:
                                msg = f"Since fails: window excludes {_short(r2_star.message)}."
                                ev = r2_star.evidence
                                span = {"start": r2_star.item, "end": now_ref} if (r2_star.item and now_ref) else None
                            else:
                                msg = "Since fails: no witness in window."
                                ev = ()
                                span = None
                            example = None
                    else:
                        sat = False
                        msg = "Since fails: no witness in window."
                        ev = ()
                        span = None
                        example = None

                    out_reason = Reason(
                        kind="since",
                        value=sat,
                        message=msg,
                        time=current_time,
                        children=tuple(x for x in (r1, r2) if x is not None),
                        item=now_ref,
                        evidence=ev,
                        span=span,
                        example_reason=example,
                    )
                    out = {"Time": current_time, "Sat": sat, "reason": out_reason}
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
        return f"Earlier({self.formula}, {self.interval})" if self.interval else f"Earlier({self.formula})"


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
        return f"Hist({self.formula}, {self.interval})" if self.interval else f"Hist({self.formula})"

    def to_term(self, name: str | None = None) -> ls.SequentialTerm:
        formula = ~Earlier(~self.formula, self.interval)
        return formula.to_term(name)
