
from pathlib import Path
import pandas as pd
import numpy as np


class FieldVisionWorkloadTrace:
    """
    Lightweight replay/sampling interface for Stage 4 task traces.
    """

    def __init__(self, csv_path, seed=42):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.index = 0

    def __len__(self):
        return len(self.df)

    def next_task(self):
        # Replay tasks in trace arrival order.
        if self.index >= len(self.df):
            return None

        row = self.df.iloc[self.index]
        self.index += 1
        return row.to_dict()

    def sample_task(self, workload=None, service_class=None):
        # Bootstrap a random task from the empirical trace.
        df = self.df

        if workload is not None:
            df = df[df["workload"] == workload]

        if service_class is not None:
            df = df[df["service_class"] == service_class]

        if len(df) == 0:
            raise ValueError("No matching tasks.")

        idx = self.rng.integers(0, len(df))
        return df.iloc[idx].to_dict()

    def tasks_arriving_between(self, start_s, end_s):
        # Return tasks whose trace arrival falls inside [start_s, end_s).
        subset = self.df[
            (self.df["arrival_time_s"] >= start_s)
            & (self.df["arrival_time_s"] < end_s)
        ]
        return subset.to_dict("records")
