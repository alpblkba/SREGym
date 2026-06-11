"""Mitigation oracle for booking_consistency_colocation_hotel_reservation.

v0.0.1 goals:

This oracle is intentionally more verbose than the minimal SREGym examples
because this fault is not meant to be solved by noticing a single broken field.
The first version verifies that the injected booking-consistency workload is
still present, still producing reports, and that the original frontend
application container has not been replaced, deleted, or repurposed.

The later hard-pass version will require a full diagnosis path. This file
already contains an env-gated trace checker for that direction:

    SREGYM_REQUIRE_DIAGNOSTIC_TRACE=1
    SREGYM_COMMAND_TRACE=/tmp/sregym_command_trace.jsonl

When enabled, the oracle requires evidence that the agent did more than a
surface-level Kubernetes check. It expects a timed frontend request, ordinary
health checks, strong low-level filesystem/kernel diagnostics, a structural
mitigation command, and repeated low-level checks after mitigation.

The trace checker is deliberately optional in v0.0.1 so we can first commit and
test the fault mechanics without making the PR depend on local CLI tracing.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kubernetes import client
from kubernetes.client.rest import ApiException

from sregym.conductor.oracles.base import Oracle


_REPORT_MAX_AGE_SECONDS = 30
_FRONTEND_PROBE_TIMEOUT_SECONDS = 5


@dataclass
class TraceEvent:
    """Small normalized view of one command-trace event.

    The local CLI tracer may evolve, so this class accepts events with slightly
    different key names. The oracle cares about semantic command categories,
    not the exact JSON schema.
    """

    timestamp: float | None
    command: str
    event_type: str
    raw: dict[str, Any]


class BookingConsistencyMitigationOracle(Oracle):
    """Verbose mitigation oracle for the booking-consistency fault."""

    importance = 1.0

    def __init__(self, problem):
        super().__init__(problem)
        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()

    def evaluate(self, solution=None, trace=None, duration=None) -> dict:
        self._section("Booking Consistency Mitigation Evaluation v0.0.1")

        namespace = self.problem.namespace
        deployment_name = self.problem.frontend_deployment

        self._log(f"namespace: {namespace}")
        self._log(f"deployment under evaluation: {deployment_name}")
        self._log(f"frontend selector: {self.problem.frontend_label_selector}")
        self._log(f"required sidecars: {self.problem.required_sidecars}")
        self._log(f"report path: {self.problem.reports_mount_path}")
        self._log(f"original ledger input path: {self.problem.original_input_path}")
        self._log(f"alternate ledger input path: {self.problem.alternate_input_path}")

        deployment = self._read_deployment(deployment_name, namespace)
        if deployment is None:
            return self._fail("frontend Deployment no longer exists")

        deployment_result = self._check_frontend_deployment_shape(deployment)
        if not deployment_result["success"]:
            return deployment_result

        pod_result = self._check_frontend_pods(namespace)
        if not pod_result["success"]:
            return pod_result

        pod_name = pod_result["pod_name"]
        report_result = self._check_report_artifacts(pod_name, namespace)
        if not report_result["success"]:
            return report_result

        service_result = self._check_frontend_service_and_endpoints(namespace)
        if not service_result["success"]:
            return service_result

        probe_result = self._check_frontend_http_probe(pod_name, namespace)
        if not probe_result["success"]:
            return probe_result

        mitigation_shape = self._classify_mitigation_shape(deployment)
        if not mitigation_shape["success"]:
            return mitigation_shape

        trace_result = self._evaluate_optional_trace_requirements()
        if not trace_result["success"]:
            return trace_result

        self._section("Oracle result")
        self._log("frontend Deployment is present")
        self._log("original frontend container is preserved")
        self._log("required booking-consistency containers are present")
        self._log("frontend pod is Running and all expected containers are Ready")
        self._log("report artifacts are fresh")
        self._log("frontend Service and Endpoints are present")
        self._log("frontend HTTP probe succeeded")
        self._log(f"mitigation classification: {mitigation_shape['classification']}")
        self._log("v0.0.1 oracle passed")

        return {
            "success": True,
            "details": {
                "pod": pod_name,
                "mitigation_classification": mitigation_shape["classification"],
                "trace": trace_result.get("summary", {}),
            },
        }

    # ------------------------------------------------------------------
    # Cluster-state checks
    # ------------------------------------------------------------------

    def _read_deployment(self, name: str, namespace: str):
        self._section("Reading frontend Deployment")
        try:
            deployment = self.apps_v1.read_namespaced_deployment(name=name, namespace=namespace)
            self._log(f"found Deployment/{name}")
            return deployment
        except ApiException as exc:
            if exc.status == 404:
                self._log(f"Deployment/{name} was not found")
                return None
            raise

    def _check_frontend_deployment_shape(self, deployment) -> dict:
        self._section("Checking frontend Deployment shape")

        desired = deployment.spec.replicas or 0
        self._log(f"desired replicas: {desired}")
        if desired < 1:
            return self._fail(f"frontend Deployment is scaled to {desired} replicas")

        pod_spec = deployment.spec.template.spec
        containers = pod_spec.containers or []
        volumes = pod_spec.volumes or []

        container_names = [container.name for container in containers]
        volume_names = [volume.name for volume in volumes]

        self._log(f"pod-template containers: {container_names}")
        self._log(f"pod-template volumes: {volume_names}")

        frontend = self._container_by_name(containers, self.problem.frontend_container_name)
        if frontend is None:
            return self._fail("original frontend container is missing")

        self._log(f"frontend container image: {frontend.image}")
        self._log(f"frontend container command: {frontend.command}")
        self._log(f"frontend container args: {frontend.args}")
        self._log(f"frontend container ports: {[p.container_port for p in frontend.ports or []]}")

        if frontend.image != self.problem.frontend_image:
            return self._fail(
                f"frontend image changed from {self.problem.frontend_image} to {frontend.image}"
            )

        if list(frontend.command or []) != list(self.problem.frontend_command):
            return self._fail(
                f"frontend command changed from {self.problem.frontend_command} to {frontend.command}"
            )

        expected_sidecars = set(self.problem.required_sidecars)
        actual_containers = set(container_names)
        missing_sidecars = sorted(expected_sidecars - actual_containers)
        if missing_sidecars:
            return self._fail(f"required booking-consistency containers are missing: {missing_sidecars}")

        for sidecar_name in self.problem.required_sidecars:
            sidecar = self._container_by_name(containers, sidecar_name)
            if sidecar is None:
                return self._fail(f"required sidecar {sidecar_name} is missing")

            self._log(f"sidecar {sidecar_name} image: {sidecar.image}")
            self._log(f"sidecar {sidecar_name} command: {sidecar.command}")
            self._log(f"sidecar {sidecar_name} resources: {sidecar.resources}")
            self._log(
                f"sidecar {sidecar_name} mounts: "
                f"{[(mount.name, mount.mount_path) for mount in sidecar.volume_mounts or []]}"
            )
            self._log(
                f"sidecar {sidecar_name} env: "
                f"{[(env.name, env.value) for env in sidecar.env or []]}"
            )

            if not self._sidecar_has_required_mounts(sidecar):
                return self._fail(f"sidecar {sidecar_name} does not mount required ledger/report volumes")

        if self.problem.ledger_volume_name not in volume_names:
            return self._fail(f"required volume {self.problem.ledger_volume_name} is missing")

        if self.problem.reports_volume_name not in volume_names:
            return self._fail(f"required volume {self.problem.reports_volume_name} is missing")

        return {"success": True}

    def _check_frontend_pods(self, namespace: str) -> dict:
        self._section("Checking frontend pods and container readiness")

        pods = self.core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=self.problem.frontend_label_selector,
        ).items

        self._log(f"number of frontend pods found: {len(pods)}")

        if not pods:
            return self._fail("no frontend pods found")

        running_ready_pods = []

        for pod in pods:
            self._log(f"pod: {pod.metadata.name}")
            self._log(f"  phase: {pod.status.phase}")
            self._log(f"  node: {pod.spec.node_name}")
            self._log(f"  pod ip: {pod.status.pod_ip}")
            self._log(f"  start time: {pod.status.start_time}")

            statuses = pod.status.container_statuses or []
            ready = {status.name for status in statuses if status.ready}
            restarts = {status.name: status.restart_count for status in statuses}

            self._log(f"  ready containers: {sorted(ready)}")
            self._log(f"  restart counts: {restarts}")

            for status in statuses:
                waiting = status.state.waiting
                terminated = status.state.terminated
                running = status.state.running
                if waiting:
                    self._log(f"  container {status.name} waiting: {waiting.reason} {waiting.message}")
                if terminated:
                    self._log(
                        f"  container {status.name} terminated: "
                        f"reason={terminated.reason} exit_code={terminated.exit_code}"
                    )
                if running:
                    self._log(f"  container {status.name} running since: {running.started_at}")

            expected = {
                self.problem.frontend_container_name,
                *self.problem.required_sidecars,
            }

            if pod.status.phase == "Running" and expected.issubset(ready):
                running_ready_pods.append(pod)

        if not running_ready_pods:
            return self._fail("no frontend pod is Running with all expected containers Ready")

        if len(running_ready_pods) > 1:
            self._log(
                "multiple ready frontend pods found; using the first one for artifact/probe checks: "
                f"{running_ready_pods[0].metadata.name}"
            )

        return {"success": True, "pod_name": running_ready_pods[0].metadata.name}

    def _check_report_artifacts(self, pod_name: str, namespace: str) -> dict:
        self._section("Checking booking-consistency report artifacts")

        report_dir = self.problem.reports_mount_path
        heartbeat = f"{report_dir}/last_successful_run"
        report_json = f"{report_dir}/booking_consistency_report.json"
        checker_json = f"{report_dir}/checker_status.json"
        reader_json = f"{report_dir}/ledger_reader_status.json"

        cmd = (
            f"kubectl exec -n {namespace} {pod_name} "
            f"-c {self.problem.report_writer_name} -- "
            f"sh -c '"
            f"echo heartbeat_path={heartbeat}; "
            f"test -s {heartbeat}; "
            f"heartbeat=$(cat {heartbeat}); "
            f"echo heartbeat=$heartbeat; "
            f"echo report_path={report_json}; "
            f"test -s {report_json}; "
            f"cat {report_json}; "
            f"echo; "
            f"echo checker_path={checker_json}; "
            f"if [ -s {checker_json} ]; then cat {checker_json}; else echo missing_checker_status; fi; "
            f"echo; "
            f"echo reader_path={reader_json}; "
            f"if [ -s {reader_json} ]; then cat {reader_json}; else echo missing_reader_status; fi"
            f"'"
        )

        output = self.problem.kubectl.exec_command(cmd).strip()
        self._log("report artifact command output:")
        self._log_block(output)

        heartbeat_ts = self._extract_heartbeat_timestamp(output)
        if heartbeat_ts is None:
            return self._fail("could not parse report heartbeat timestamp")

        age = int(time.time()) - heartbeat_ts
        self._log(f"report heartbeat age: {age}s")

        if age < 0:
            return self._fail("report heartbeat timestamp is in the future")

        if age > _REPORT_MAX_AGE_SECONDS:
            return self._fail(
                f"report heartbeat is stale: age={age}s threshold={_REPORT_MAX_AGE_SECONDS}s"
            )

        return {"success": True}

    def _check_frontend_service_and_endpoints(self, namespace: str) -> dict:
        self._section("Checking frontend Service and Endpoints")

        try:
            service = self.core_v1.read_namespaced_service("frontend", namespace)
        except ApiException as exc:
            if exc.status == 404:
                return self._fail("frontend Service no longer exists")
            raise

        selector = service.spec.selector or {}
        ports = [(port.name, port.port, port.target_port) for port in service.spec.ports or []]
        self._log(f"frontend Service selector: {selector}")
        self._log(f"frontend Service ports: {ports}")

        expected_selector_value = "frontend"
        if selector.get("io.kompose.service") != expected_selector_value:
            return self._fail(
                "frontend Service selector changed unexpectedly: "
                f"io.kompose.service={selector.get('io.kompose.service')}"
            )

        try:
            endpoints = self.core_v1.read_namespaced_endpoints("frontend", namespace)
        except ApiException as exc:
            if exc.status == 404:
                return self._fail("frontend Endpoints object no longer exists")
            raise

        addresses = []
        not_ready = []
        for subset in endpoints.subsets or []:
            addresses.extend([addr.ip for addr in subset.addresses or []])
            not_ready.extend([addr.ip for addr in subset.not_ready_addresses or []])

        self._log(f"frontend ready endpoint addresses: {addresses}")
        self._log(f"frontend not-ready endpoint addresses: {not_ready}")

        if not addresses:
            return self._fail("frontend Service has no ready endpoint addresses")

        return {"success": True}

    def _check_frontend_http_probe(self, pod_name: str, namespace: str) -> dict:
        self._section("Checking frontend HTTP probe")

        # Use a sidecar container in the same pod because busybox has wget and
        # the pod network namespace can reach the frontend container through
        # localhost. This avoids depending on curl/wget inside the original
        # frontend image.
        cmd = (
            f"kubectl exec -n {namespace} {pod_name} "
            f"-c {self.problem.report_writer_name} -- "
            f"sh -c 'wget -q -T {_FRONTEND_PROBE_TIMEOUT_SECONDS} "
            f"-O /dev/null http://127.0.0.1:5000/ && echo frontend_probe_ok'"
        )

        output = self.problem.kubectl.exec_command(cmd).strip()
        self._log(f"frontend probe output: {output!r}")

        if "frontend_probe_ok" not in output:
            return self._fail("frontend HTTP probe from the pod did not succeed")

        return {"success": True}

    def _classify_mitigation_shape(self, deployment) -> dict:
        self._section("Classifying mitigation shape")

        containers = deployment.spec.template.spec.containers or []
        ledger_reader = self._container_by_name(containers, self.problem.ledger_reader_name)
        if ledger_reader is None:
            return self._fail("ledger-reader sidecar is missing")

        env = {item.name: item.value for item in ledger_reader.env or []}
        self._log(f"ledger-reader env: {env}")

        input_path = env.get(self.problem.input_env)
        interval = self._safe_int(env.get(self.problem.interval_env))
        file_count = self._safe_int(env.get(self.problem.file_count_env))
        file_mib = self._safe_int(env.get(self.problem.file_mib_env))

        original_interval = self._safe_int(self.problem.default_interval_seconds)
        original_file_count = self._safe_int(self.problem.default_file_count)
        original_file_mib = self._safe_int(self.problem.default_file_mib)

        self._log(f"ledger input path: {input_path}")
        self._log(f"ledger interval: {interval} original={original_interval}")
        self._log(f"ledger file_count: {file_count} original={original_file_count}")
        self._log(f"ledger file_mib: {file_mib} original={original_file_mib}")

        # v0.0.1 accepts the initial injected state so we can smoke-test the
        # problem. Once we start validating real agent fixes, this method will
        # become stricter and require one of the accepted mitigation shapes.
        if input_path == self.problem.original_input_path:
            self._log("ledger-reader is still using the original input path")
            self._log("v0.0.1 accepts this for smoke testing; hard-pass will require a real mitigation")
            return {"success": True, "classification": "unmitigated-smoke-test-state"}

        if input_path == self.problem.alternate_input_path:
            return {"success": True, "classification": "input-path-changed"}

        if interval is not None and original_interval is not None and interval > original_interval:
            return {"success": True, "classification": "interval-increased"}

        if (
            file_count is not None
            and original_file_count is not None
            and file_mib is not None
            and original_file_mib is not None
            and file_count * file_mib < original_file_count * original_file_mib
        ):
            return {"success": True, "classification": "working-set-bounded"}

        return self._fail("ledger-reader changed, but not in a recognized safe mitigation shape")

    # ------------------------------------------------------------------
    # Optional trace-aware checks
    # ------------------------------------------------------------------

    def _evaluate_optional_trace_requirements(self) -> dict:
        self._section("Evaluating optional diagnostic trace")

        require_trace = os.environ.get("SREGYM_REQUIRE_DIAGNOSTIC_TRACE") == "1"
        trace_path = Path(os.environ.get("SREGYM_COMMAND_TRACE", "/tmp/sregym_command_trace.jsonl"))

        self._log(f"SREGYM_REQUIRE_DIAGNOSTIC_TRACE={require_trace}")
        self._log(f"SREGYM_COMMAND_TRACE={trace_path}")

        if not trace_path.exists():
            if require_trace:
                return self._fail(f"diagnostic trace is required but missing: {trace_path}")
            self._log("diagnostic trace not found; skipping trace requirements in v0.0.1")
            return {"success": True, "summary": {"trace_present": False}}

        events = self._load_trace_events(trace_path)
        self._log(f"loaded trace events: {len(events)}")

        summary = self._summarize_trace(events)
        self._log(f"trace summary: {summary}")

        if not require_trace:
            self._log("trace is present but not required; summary only")
            return {"success": True, "summary": summary}

        missing = []

        if not summary["timed_frontend_request"]:
            missing.append("timed frontend request evidence")

        if not summary["ordinary_health_checks"]:
            missing.append("ordinary Kubernetes/app health checks")

        if not summary["same_pod_inspection"]:
            missing.append("same-pod/container inspection")

        if not summary["strong_low_level_before_mitigation"]:
            missing.append("strong low-level diagnostic before mitigation")

        if not summary["structural_mitigation_command"]:
            missing.append("structural mitigation command")

        if not summary["strong_low_level_after_30s"]:
            missing.append("strong low-level diagnostic at least 30s after mitigation")

        if not summary["strong_low_level_after_60s"]:
            missing.append("strong low-level diagnostic at least 60s after mitigation")

        if missing:
            return self._fail(
                "hard-pass diagnostic trace requirements were not met: " + ", ".join(missing)
            )

        return {"success": True, "summary": summary}

    def _load_trace_events(self, trace_path: Path) -> list[TraceEvent]:
        events = []
        for line in trace_path.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            command = self._extract_command_from_trace_event(raw)
            event_type = str(raw.get("event") or raw.get("type") or raw.get("event_type") or "")
            timestamp = self._extract_timestamp_from_trace_event(raw)

            events.append(
                TraceEvent(
                    timestamp=timestamp,
                    command=command,
                    event_type=event_type,
                    raw=raw,
                )
            )
        return events

    def _summarize_trace(self, events: list[TraceEvent]) -> dict[str, Any]:
        commands = [event for event in events if event.command]
        mitigation_time = self._first_event_time(commands, self._is_structural_mitigation_command)

        strong_before = False
        strong_after_30 = False
        strong_after_60 = False

        for event in commands:
            if not self._is_strong_low_level_command(event.command):
                continue

            if mitigation_time is None or event.timestamp is None:
                if mitigation_time is None:
                    strong_before = True
                continue

            delta = event.timestamp - mitigation_time
            if delta < 0:
                strong_before = True
            if delta >= 30:
                strong_after_30 = True
            if delta >= 60:
                strong_after_60 = True

        return {
            "trace_present": True,
            "command_events": len(commands),
            "timed_frontend_request": any(self._is_timed_frontend_request(e.command) for e in commands),
            "ordinary_health_checks": any(self._is_ordinary_health_check(e.command) for e in commands),
            "same_pod_inspection": any(self._is_same_pod_inspection(e.command) for e in commands),
            "filesystem_inspection": any(self._is_filesystem_inspection(e.command) for e in commands),
            "strong_low_level_command_seen": any(self._is_strong_low_level_command(e.command) for e in commands),
            "structural_mitigation_command": mitigation_time is not None,
            "strong_low_level_before_mitigation": strong_before,
            "strong_low_level_after_30s": strong_after_30,
            "strong_low_level_after_60s": strong_after_60,
        }

    # ------------------------------------------------------------------
    # Trace classifiers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_timed_frontend_request(command: str) -> bool:
        lowered = command.lower()
        return (
            ("curl" in lowered or "wget" in lowered or "time " in lowered)
            and ("frontend" in lowered or "127.0.0.1:5000" in lowered or ":5000" in lowered)
            and ("time_total" in lowered or "time " in lowered or "seq " in lowered or "for " in lowered)
        )

    @staticmethod
    def _is_ordinary_health_check(command: str) -> bool:
        lowered = command.lower()
        patterns = [
            "kubectl get pod",
            "kubectl get pods",
            "kubectl describe pod",
            "kubectl get svc",
            "kubectl get service",
            "kubectl get endpoints",
            "kubectl get deployment",
            "kubectl describe deployment",
            "kubectl logs",
            "kubectl top",
        ]
        return any(pattern in lowered for pattern in patterns)

    @staticmethod
    def _is_same_pod_inspection(command: str) -> bool:
        lowered = command.lower()
        return (
            "kubectl exec" in lowered
            and (
                " ps" in lowered
                or "cat /proc" in lowered
                or "mount" in lowered
                or "df " in lowered
                or "du " in lowered
                or "find " in lowered
                or "ls " in lowered
                or "/var/lib/booking" in lowered
            )
        )

    @staticmethod
    def _is_filesystem_inspection(command: str) -> bool:
        lowered = command.lower()
        patterns = [
            " df ",
            " du ",
            " mount",
            " find ",
            " lsof",
            " stat ",
            " ls ",
            "/var/lib/booking",
        ]
        return any(pattern in lowered for pattern in patterns)

    @staticmethod
    def _is_strong_low_level_command(command: str) -> bool:
        lowered = command.lower()
        patterns = [
            "/proc/vmstat",
            "/proc/pressure/io",
            "/proc/pressure/memory",
            "/sys/fs/cgroup",
            "memory.stat",
            "io.stat",
            "vmstat",
            "iostat",
            "pidstat",
            "sar -b",
            "sar -d",
            "workingset_refault",
            "pgmajfault",
            "pgscan",
            "pgsteal",
        ]
        return any(pattern in lowered for pattern in patterns)

    @staticmethod
    def _is_structural_mitigation_command(command: str) -> bool:
        lowered = command.lower()
        structural_words = [
            "kubectl patch",
            "kubectl edit",
            "kubectl apply",
            "kubectl replace",
            "kubectl set env",
        ]
        target_words = [
            "booking_ledger_input_path",
            "booking_ledger_interval_seconds",
            "booking_ledger_file_count",
            "booking_ledger_file_mib",
            "ledger-reader",
            "frontend",
        ]
        return any(word in lowered for word in structural_words) and any(
            word in lowered for word in target_words
        )

    # ------------------------------------------------------------------
    # Small utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _container_by_name(containers, name: str):
        for container in containers:
            if container.name == name:
                return container
        return None

    def _sidecar_has_required_mounts(self, sidecar) -> bool:
        mounts = {mount.name: mount.mount_path for mount in sidecar.volume_mounts or []}
        return (
            mounts.get(self.problem.ledger_volume_name) == self.problem.ledger_mount_path
            and mounts.get(self.problem.reports_volume_name) == self.problem.reports_mount_path
        )

    @staticmethod
    def _extract_heartbeat_timestamp(output: str) -> int | None:
        for line in output.splitlines():
            if line.startswith("heartbeat="):
                raw = line.split("=", 1)[1].strip()
                try:
                    return int(raw)
                except ValueError:
                    return None
        return None

    @staticmethod
    def _safe_int(value) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_command_from_trace_event(raw: dict[str, Any]) -> str:
        for key in ("command", "raw_command", "input", "cmd", "text"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        for nested_key in ("data", "payload", "details"):
            nested = raw.get(nested_key)
            if isinstance(nested, dict):
                for key in ("command", "raw_command", "input", "cmd", "text"):
                    value = nested.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()

        return ""

    @staticmethod
    def _extract_timestamp_from_trace_event(raw: dict[str, Any]) -> float | None:
        for key in ("timestamp", "time", "ts", "monotonic_time"):
            value = raw.get(key)
            if isinstance(value, (int, float)):
                return float(value)

        return None

    @staticmethod
    def _first_event_time(events: list[TraceEvent], predicate) -> float | None:
        for event in events:
            if event.timestamp is not None and predicate(event.command):
                return event.timestamp
        return None

    @staticmethod
    def _section(title: str):
        print(f"\n== {title} ==")

    @staticmethod
    def _log(message: str):
        print(f"[booking-consistency-oracle] {message}")

    @classmethod
    def _log_block(cls, text: str):
        if not text:
            cls._log("<empty>")
            return
        for line in text.splitlines():
            cls._log(f"  {line}")

    @classmethod
    def _fail(cls, reason: str) -> dict:
        cls._section("Oracle failure")
        cls._log(f"❌ {reason}")
        return {"success": False, "reason": reason}
