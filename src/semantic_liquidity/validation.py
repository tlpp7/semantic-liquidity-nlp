"""P4: reproducible stratified partitions and paired uncertainty estimates."""
from collections import defaultdict
import math
import random
import statistics

def stratified_split(rows, seed=42, fractions=(0.8, 0.1, 0.1)):
    if len(fractions) != 3 or any(x <= 0 for x in fractions) or not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
        raise ValueError("Fractions must be positive and sum to one")
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["label"])].append(row)
    if len(groups) < 2 or any(len(items) < 10 for items in groups.values()):
        raise ValueError("Stratification requires at least 10 samples per class")
    rng = random.Random(seed)
    splits = [[], [], []]
    for group in groups.values():
        shuffled = list(group)
        rng.shuffle(shuffled)
        n = len(shuffled)
        a = max(1, int(n * fractions[0]))
        b = max(a+1, int(n * (fractions[0] + fractions[1])))
        if b >= n:
            b = n-1
        for part, chunk in zip(splits, (shuffled[:a], shuffled[a:b], shuffled[b:])):
            part.extend(chunk)
    for part in splits:
        rng.shuffle(part)
    return tuple(splits)

def paired_bootstrap(a, b, repeats=2000, seed=42, confidence=0.95):
    """Paired bootstrap CI for mean(a-b), preserving paired observations."""
    if len(a) != len(b) or not a or repeats < 100 or not (0 < confidence < 1):
        raise ValueError("Require equally sized nonempty vectors, repeats>=100, confidence in (0,1)")
    diffs = [float(x)-float(y) for x,y in zip(a,b)]
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(repeats))
    alpha = (1-confidence)/2
    lo = means[min(len(means)-1, int(alpha*len(means)))]
    hi = means[min(len(means)-1, int((1-alpha)*len(means)))]
    return {"mean_difference":statistics.mean(diffs), "ci_low":lo, "ci_high":hi,
            "confidence":confidence, "n_pairs":len(diffs), "method":"paired percentile bootstrap"}

def detect_exact_duplicates(rows):
    """Count exact normalized text duplicates and conflicting labels."""
    seen = {}
    duplicate_count = 0
    conflicting_count = 0
    for row in rows:
        key = " ".join(str(row["text"]).casefold().split())
        label = str(row["label"])
        if key in seen:
            duplicate_count += 1
            if label != seen[key]:
                conflicting_count += 1
        else:
            seen[key] = label
    return {"duplicates":duplicate_count, "conflicting_labels":conflicting_count}

def leakage_report(train, validation, test):
    sets = [set(" ".join(str(r["text"]).casefold().split()) for r in split) for split in (train,validation,test)]
    return {"train_val_overlap":len(sets[0]&sets[1]), "train_test_overlap":len(sets[0]&sets[2]),
            "val_test_overlap":len(sets[1]&sets[2])}
