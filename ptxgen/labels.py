"""Branch label naming for one kernel."""

from collections import defaultdict


class LabelAllocator:
    """Hands out unique branch label names for one kernel, such as LOOP_0 then LOOP_1."""

    def __init__(self):
        self.counter = defaultdict(int)

    def new_label(self, prefix: str) -> str:
        """Return a fresh label name with the given prefix."""
        label_id = self.counter[prefix]
        self.counter[prefix] += 1
        return f"{prefix}_{label_id}"
