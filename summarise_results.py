import os
from typing import Any

import numpy as np
import pandas as pd

OUTPUT_ROOT = "out"


def list_models() -> list[str]:
    return [
        nm
        for nm in os.listdir(OUTPUT_ROOT)
        if os.path.isdir(os.path.join(OUTPUT_ROOT, nm))
    ]


def get_times(model: str) -> np.ndarray:
    with open(os.path.join(OUTPUT_ROOT, model, "times.txt"), "r") as f:
        return np.array([float(line) * 1000 for line in f.readlines()])


def get_cpu(model: str) -> np.ndarray:
    with open(os.path.join(OUTPUT_ROOT, model, "cpu.txt"), "r") as f:
        return np.array([float(line) for line in f.readlines()])


def calculate_statistics(times: np.ndarray) -> dict[str, Any]:
    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "P95": np.percentile(times, 95),
        "P98": np.percentile(times, 98),
        "P99": np.percentile(times, 99),
    }


if __name__ == "__main__":
    models = list_models()
    results = []
    results_cpu = []
    for model in models:
        times = get_times(model)
        stats = calculate_statistics(times)
        results.append(stats)
        cpu = get_cpu(model)
        stats_cpu = calculate_statistics(cpu)
        results_cpu.append(stats_cpu)
    results_pd = pd.DataFrame(results, index=models)
    results_cpu_pd = pd.DataFrame(results_cpu, index=models)

    results_pd.to_csv(os.path.join(OUTPUT_ROOT, "results_time.csv"), index=True)
    results_cpu_pd.to_csv(os.path.join(OUTPUT_ROOT, "results_cpu.csv"), index=True)
