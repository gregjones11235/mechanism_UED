import json
import os
import threading
import time
import uuid
import contextvars
import csv
import tempfile
from collections import defaultdict
from contextlib import contextmanager

import matplotlib.pyplot as plt
import pandas as pd


_EVENT_FIELDS = (
    "run_id", "session", "phase", "parent_phase", "start_monotonic_ns",
    "end_monotonic_ns", "duration_s", "status", "cache_hit",
    "task_signature", "request_id", "overlap_group",
)


class RuntimeTracker:
	_instance = None
	_lock = threading.Lock()

	def __new__(cls, *args, **kwargs):
		if not cls._instance:
			with cls._lock:
				if not cls._instance:
					cls._instance = super(RuntimeTracker, cls).__new__(cls)
					cls._instance._initialized = False
		return cls._instance

	def __init__(self, output_dir="runtime_analysis"):
		if self._initialized:
			return
		self.output_dir = output_dir
		self.timings = []
		self.current_timers = {}
		self.lock = threading.RLock()
		self.enabled = False
		self.output_jsonl = os.path.join(output_dir, "events.jsonl")
		self.run_id = uuid.uuid4().hex
		self._session = contextvars.ContextVar("runtime_session", default=None)
		self._phase_stack = contextvars.ContextVar("runtime_phase_stack", default=())
		self._initialized = True

	def configure(self, config=None, *, enabled=None, output_jsonl=None, run_id=None, reset=False):
		"""Configure event profiling; accepts a Hydra config or explicit values."""
		if config is not None:
			section = config.get("runtime_profiling", {}) if hasattr(config, "get") else {}
			enabled = section.get("enabled", False) if enabled is None else enabled
			output_jsonl = section.get("output_jsonl", "runtime_analysis/events.jsonl") if output_jsonl is None else output_jsonl
		with self.lock:
			if reset:
				self.run_id = str(run_id or uuid.uuid4().hex)
				self._session.set(None)
				self._phase_stack.set(())
				self.current_timers.clear()
				self.timings.clear()
			self.enabled = bool(False if enabled is None else enabled)
			if output_jsonl:
				self.output_jsonl = os.fspath(output_jsonl)
				self.output_dir = os.path.dirname(self.output_jsonl) or self.output_dir
			if run_id:
				self.run_id = str(run_id)
		return self

	def set_session(self, session):
		self._session.set(session)

	@contextmanager
	def session(self, session):
		token = self._session.set(session)
		try:
			yield
		finally:
			self._session.reset(token)

	def _append_event(self, event):
		if not self.enabled:
			return
		row = {key: event.get(key) for key in _EVENT_FIELDS}
		with self.lock:
			parent = os.path.dirname(self.output_jsonl)
			if parent:
				os.makedirs(parent, exist_ok=True)
			with open(self.output_jsonl, "a", encoding="utf-8") as handle:
				handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")

	def record(self, phase, start_monotonic_ns=None, end_monotonic_ns=None,
			   session=None, parent_phase=None, status="ok", cache_hit=False,
			   task_signature=None, request_id=None, overlap_group=None):
		if session is None:
			session = self._session.get()
		if parent_phase is None:
			stack = self._phase_stack.get()
			parent_phase = stack[-1] if stack else None
		start = time.monotonic_ns() if start_monotonic_ns is None else int(start_monotonic_ns)
		end = time.monotonic_ns() if end_monotonic_ns is None else int(end_monotonic_ns)
		event = {"run_id": self.run_id, "session": session, "phase": phase,
				 "parent_phase": parent_phase, "start_monotonic_ns": start,
				 "end_monotonic_ns": end, "duration_s": max(0, end - start) / 1e9,
				 "status": status, "cache_hit": bool(cache_hit),
				 "task_signature": task_signature, "request_id": request_id,
				 "overlap_group": overlap_group}
		self._append_event(event)
		return event

	@contextmanager
	def span(self, phase, **kwargs):
		start = time.monotonic_ns()
		requested_status = kwargs.pop("status", "ok")
		stack = self._phase_stack.get()
		if "parent_phase" not in kwargs:
			kwargs["parent_phase"] = stack[-1] if stack else None
		token = self._phase_stack.set(stack + (phase,))
		status = requested_status
		try:
			yield
		except Exception:
			status = "error"
			raise
		finally:
			self._phase_stack.reset(token)
			self.record(phase, start, status=status, **kwargs)

	def start_timer(self, component_name):
		with self.lock:
			self.current_timers[component_name] = (time.time(), time.monotonic_ns())

	def stop_timer(self, component_name, session_idx):
		with self.lock:
			if component_name not in self.current_timers:
				return 0.0
			start_time, start_mono = self.current_timers.pop(component_name)
			duration = time.time() - start_time
			self.timings.append({"session": session_idx, "component": component_name, "duration": duration})
		self.record(component_name, session=session_idx, end_monotonic_ns=time.monotonic_ns(),
					start_monotonic_ns=start_mono)
		return duration

	def log_duration(self, component_name, session_idx, duration):
		with self.lock:
			self.timings.append({"session": session_idx, "component": component_name, "duration": duration})
		if self.enabled:
			end = time.monotonic_ns()
			self.record(component_name, end - int(float(duration) * 1e9), end, session=session_idx)

	def save_data(self):
		with self.lock:
			if not self.timings:
				return
			os.makedirs(self.output_dir, exist_ok=True)
			pd.DataFrame(self.timings).to_csv(os.path.join(self.output_dir, "timings.csv"), index=False)

	def derive_reports(self):
		"""Derive profiling reports from JSONL using atomic writes.

		The derivation is intentionally a no-op while profiling is disabled so a
		baseline run cannot create profiling artifacts.
		"""
		if not self.enabled or not os.path.exists(self.output_jsonl):
			return None
		with self.lock:
			with open(self.output_jsonl, "r", encoding="utf-8") as handle:
				events = [json.loads(line) for line in handle if line.strip()]
		# A resumed run may append to an existing JSONL.  Reports are scoped to
		# this tracker instance and never mix records from another run_id.
		events = [event for event in events if event.get("run_id") == self.run_id]
		if not events:
			return None
		csv_path = os.path.join(self.output_dir, "events.csv")
		json_path = os.path.join(self.output_dir, "critical_path.json")
		os.makedirs(self.output_dir, exist_ok=True)
		self._atomic_csv(csv_path, events)
		sessions = defaultdict(list)
		for event in events:
			if event.get("session") is not None:
				sessions[str(event.get("session"))].append(event)
		phase_totals = defaultdict(float)
		exclusive_phase_totals = defaultdict(float)
		session_reports = {}
		for session, rows in sessions.items():
			wall_rows = [r for r in rows if r.get("phase") == "session_wall"]
			if wall_rows:
				wall_start = min(int(r["start_monotonic_ns"]) for r in wall_rows)
				wall_end = max(int(r["end_monotonic_ns"]) for r in wall_rows)
			else:
				wall_start = min(int(r["start_monotonic_ns"]) for r in rows)
				wall_end = max(int(r["end_monotonic_ns"]) for r in rows)
			def clipped(row):
				start = max(wall_start, int(row["start_monotonic_ns"]))
				end = min(wall_end, int(row["end_monotonic_ns"]))
				return (start, end) if end >= start else None
			work_rows = []
			for row in rows:
				if row.get("phase") == "session_wall":
					continue
				interval = clipped(row)
				if interval is not None:
					work_rows.append((row, interval[0], interval[1]))
			intervals = [(start, end) for _, start, end in work_rows]
			covered_ns = self._union_ns(intervals)
			overlap_groups = defaultdict(list)
			for row, start, end in work_rows:
				overlap_groups[str(row.get("overlap_group") or "__ungrouped__")].append((start, end))
			phase_report = {}
			for phase in sorted({r.get("phase") for r in rows}):
				phase_intervals = [(start, end) for row, start, end in work_rows if row.get("phase") == phase]
				duration = self._union_ns(phase_intervals) / 1e9
				phase_report[phase] = duration
				phase_totals[phase] += duration
			# Attribute each timeline segment to one deepest active phase.  This
			# removes parent/child and concurrent overlap double-counting while
			# retaining inclusive totals above.
			exclusive_ns = defaultdict(int)
			boundaries = sorted({point for _, start, end in work_rows for point in (start, end)})
			def depth(row, seen=None):
				seen = set() if seen is None else seen
				phase = row.get("phase")
				if phase in seen or not row.get("parent_phase"):
					return 0
				seen.add(phase)
				parent = next((candidate for candidate, _, _ in work_rows
				               if candidate.get("phase") == row.get("parent_phase")), None)
				return 1 + depth(parent, seen) if parent else 0
			for left, right in zip(boundaries, boundaries[1:]):
				active = [row for row, start, end in work_rows if start <= left and end >= right]
				if active and right > left:
					chosen = max(active, key=lambda row: (depth(row), int(row.get("start_monotonic_ns", 0))))
					exclusive_ns[str(chosen.get("phase"))] += right - left
			exclusive_report = {phase: value / 1e9 for phase, value in exclusive_ns.items()}
			for phase, duration in exclusive_report.items():
				exclusive_phase_totals[phase] += duration
			wall_s = max(0, wall_end - wall_start) / 1e9
			session_reports[session] = {
				"session_wall": wall_s,
				"covered_union": covered_ns / 1e9,
				"unattributed": max(0.0, wall_s - covered_ns / 1e9),
				"phase_totals": phase_report,
				"exclusive_phase_totals": exclusive_report,
				"overlap_groups": {group: {"covered_union_s": self._union_ns(group_intervals) / 1e9}
				                   for group, group_intervals in overlap_groups.items()},
			}
		report = {"run_id": self.run_id, "sessions": dict(session_reports),
		          "phase_totals": dict(phase_totals),
		          "exclusive_phase_totals": dict(exclusive_phase_totals),
		          "session_wall": sum(item["session_wall"] for item in session_reports.values()),
		          "covered_union": sum(item["covered_union"] for item in session_reports.values()),
		          "unattributed": sum(item["unattributed"] for item in session_reports.values()),
		          "critical_path": sorted(
				  ({"phase": phase, "duration_s": duration} for phase, duration in exclusive_phase_totals.items()),
				  key=lambda item: item["duration_s"], reverse=True),
		          }
		self._atomic_json(json_path, report)
		return report

	# Backwards/forwards-compatible names used by runners and tests.
	write_reports = derive_reports
	finalize_reports = derive_reports
	export_reports = derive_reports
	generate_reports = derive_reports

	@staticmethod
	def _union_ns(intervals):
		merged = []
		for start, end in sorted((a, b) for a, b in intervals if b >= a):
			if merged and start <= merged[-1][1]:
				merged[-1] = (merged[-1][0], max(merged[-1][1], end))
			else:
				merged.append((start, end))
		return sum(end - start for start, end in merged)

	@staticmethod
	def _atomic_json(path, payload):
		directory = os.path.dirname(path) or "."
		fd, temp_path = tempfile.mkstemp(prefix=".runtime-", suffix=".tmp", dir=directory)
		try:
			with os.fdopen(fd, "w", encoding="utf-8") as handle:
				json.dump(payload, handle, sort_keys=True, indent=2)
				handle.flush()
				os.fsync(handle.fileno())
			os.replace(temp_path, path)
		finally:
			if os.path.exists(temp_path):
				os.unlink(temp_path)

	@staticmethod
	def _atomic_csv(path, rows):
		directory = os.path.dirname(path) or "."
		fd, temp_path = tempfile.mkstemp(prefix=".runtime-", suffix=".tmp", dir=directory)
		try:
			with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
				writer = csv.DictWriter(handle, fieldnames=_EVENT_FIELDS)
				writer.writeheader()
				writer.writerows({key: row.get(key) for key in _EVENT_FIELDS} for row in rows)
				handle.flush()
				os.fsync(handle.fileno())
			os.replace(temp_path, path)
		finally:
			if os.path.exists(temp_path):
				os.unlink(temp_path)

	def plot_results(self):
		with self.lock:
			if not self.timings:
				return
			df = pd.DataFrame(self.timings)
		if df.empty:
			return
		pivot_df = df.pivot_table(index="session", columns="component", values="duration", aggfunc="sum")
		ax = pivot_df.plot(kind="bar", stacked=False)
		ax.set_title("Runtime Breakdown per Session")
		ax.set_xlabel("Session Index")
		ax.set_ylabel("Time (seconds)")
		ax.legend(title="Component")
		plt.tight_layout()
		os.makedirs(self.output_dir, exist_ok=True)
		plt.savefig(os.path.join(self.output_dir, "runtime_breakdown.png"))
		plt.close()


tracker = RuntimeTracker()
