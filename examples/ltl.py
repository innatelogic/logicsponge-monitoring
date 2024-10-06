import random
import time
from datetime import datetime, timedelta

import pytz

import datasponge.core as ds
from datasponge.core import dashboard
from datasponge.monitoring import Earlier, Previous, Proposition, Time, TimeInterval


def generate_random_dict(current_time: Time | None = None) -> ds.DataItem:
    if current_time is None:
        return ds.DataItem({"A": bool(random.getrandbits(1)), "B": bool(random.getrandbits(1))})
    return ds.DataItem(
        {
            "Time": current_time,
            "A": bool(random.getrandbits(1)),
            "B": bool(random.getrandbits(1)),
        }
    )


class Source(ds.SourceTerm):
    current_time: datetime = datetime.now(pytz.timezone("Europe/Paris"))

    def run(self):
        while True:
            time.sleep(1)
            out = generate_random_dict(self.current_time)  # generates random dict with datetime keys
            self.current_time += timedelta(seconds=5)
            self.output(out)


def main():
    # MTL formulas
    sub_formula = Proposition(lambda data: data["B"]) & Earlier(
        Proposition(lambda data: data["A"]),
        TimeInterval(timedelta(seconds=5), timedelta(seconds=5)),
    )
    formula = Earlier(sub_formula, TimeInterval(timedelta(seconds=5), timedelta(seconds=5)))

    # Circuit for monitoring the formula
    monitor = formula.to_term()
    monitor_p = Previous(formula).to_term()

    circuit = (
        Source()
        * (ds.KeyFilter(not_keys="Time") | monitor | monitor_p * ds.Rename(fun=lambda x: "P" if x == "Sat" else x))
        * ds.ToSingleStream(merge=True)
        * dashboard.BinaryPlot(x="Time", y=["A", "B", "Sat", "P"])
    )
    circuit.start()

    dashboard.show_stats(circuit)
    dashboard.run()


if __name__ == "__main__":
    main()
