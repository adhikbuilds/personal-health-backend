from __future__ import annotations

"""
Personal Health — minimal Prometheus text exposition.

Hand-rolled, no prometheus_client dep. Counters, gauges, and a histogram
for HTTP request duration. Thread-safe via simple locks.
"""

import threading
from collections import defaultdict

_LOCK = threading.Lock()

# counter: name -> labels-tuple -> value
_counters: dict[str, dict[tuple, float]] = defaultdict(lambda: defaultdict(float))
# gauge: name -> value
_gauges: dict[str, float] = defaultdict(float)
# histogram: name -> labels-tuple -> {sum, count, buckets}
_HIST_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
_histograms: dict[str, dict[tuple, dict]] = defaultdict(
    lambda: defaultdict(lambda: {"sum": 0.0, "count": 0, "buckets": [0] * len(_HIST_BUCKETS)})
)


def inc_counter(name: str, labels: dict | None = None, value: float = 1.0) -> None:
    key = tuple(sorted((labels or {}).items()))
    with _LOCK:
        _counters[name][key] += value


def set_gauge(name: str, value: float) -> None:
    with _LOCK:
        _gauges[name] = value


def observe_histogram(name: str, value: float, labels: dict | None = None) -> None:
    key = tuple(sorted((labels or {}).items()))
    with _LOCK:
        h = _histograms[name][key]
        h["sum"] += value
        h["count"] += 1
        for i, bound in enumerate(_HIST_BUCKETS):
            if value <= bound:
                h["buckets"][i] += 1


def observe_request(method: str, path: str, status: int, duration_seconds: float) -> None:
    # Collapse high-cardinality paths to route templates would require ASGI hooks;
    # for now we group by first segment to keep cardinality bounded.
    route = "/" + path.lstrip("/").split("/")[0] if path != "/" else "/"
    labels = {"route": route, "method": method, "status": str(status)}
    inc_counter("http_requests_total", labels)
    observe_histogram("http_request_duration_seconds", duration_seconds, {"route": route, "method": method})


def _format_labels(labels_tuple: tuple) -> str:
    if not labels_tuple:
        return ""
    parts = ",".join(f'{k}="{v}"' for k, v in labels_tuple)
    return "{" + parts + "}"


def render() -> str:
    """Return Prometheus text exposition format."""
    with _LOCK:
        lines: list[str] = []
        for name, by_labels in _counters.items():
            lines.append(f"# TYPE {name} counter")
            for labels, value in by_labels.items():
                lines.append(f"{name}{_format_labels(labels)} {value}")
        for name, value in _gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        for name, by_labels in _histograms.items():
            lines.append(f"# TYPE {name} histogram")
            for labels, h in by_labels.items():
                base_labels = list(labels)
                cumulative = 0
                for i, bound in enumerate(_HIST_BUCKETS):
                    cumulative += h["buckets"][i]
                    bucket_labels = tuple([*base_labels, ("le", str(bound))])
                    lines.append(f"{name}_bucket{_format_labels(bucket_labels)} {cumulative}")
                inf_labels = tuple([*base_labels, ("le", "+Inf")])
                lines.append(f"{name}_bucket{_format_labels(inf_labels)} {h['count']}")
                lines.append(f"{name}_sum{_format_labels(labels)} {h['sum']}")
                lines.append(f"{name}_count{_format_labels(labels)} {h['count']}")
        return "\n".join(lines) + "\n"
