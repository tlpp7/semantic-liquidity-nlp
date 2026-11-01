"""P8: deterministic bounded rehearsal buffer for continual learning."""
import random
from collections import Counter

class ReplayBuffer:
    def __init__(self, capacity=256, seed=42):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.rows = []
        self.seen = 0

    def add(self, records):
        """Reservoir sampling: uniform over all observations seen so far."""
        for record in records:
            self.seen += 1
            if len(self.rows) < self.capacity:
                self.rows.append(dict(record))
            else:
                idx = self.rng.randrange(self.seen)
                if idx < self.capacity:
                    self.rows[idx] = dict(record)

    def sample(self, count):
        if count < 0:
            raise ValueError("count must be nonnegative")
        return [dict(row) for row in self.rng.sample(self.rows, min(count,len(self.rows)))]

    def stats(self):
        return {"seen":self.seen,"stored":len(self.rows),
                "label_counts":dict(Counter(str(r["label"]) for r in self.rows))}
