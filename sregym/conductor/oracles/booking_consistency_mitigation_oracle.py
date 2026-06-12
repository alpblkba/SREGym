"""Mitigation oracle for booking_consistency_colocation_hotel_reservation.

The fault injects required booking-consistency sidecars into the Hotel
Reservation frontend pod. The report-writer sidecar performs live Mongo-backed
booking diagnostics. A valid mitigation must preserve the diagnostics/reporting
path while reducing the live validation pressure enough for the diagnostics
report to become fresh and healthy again.
"""

from __future__ import annotations

import json
import time
from typing import Any

from kubernetes import client
from kubernetes.client.rest import ApiException

from sregym.conductor.oracles.base import Oracle


_REPORT_SAMPLE_COUNT = 3
_REPORT_SAMPLE_INTERVAL_SECONDS = 10

_MAX_LAST_SUCCESS_AGE_SECONDS = 20
_MAX_ERROR_RATE = 0.005
_MAX_P95_MS = 500
_MAX_P99_MS = 1000

_MAX_ACCEPTED_DIAGNOSTIC_WORKERS = 32
_MAX_ACCEPTED_DIAGNOSTIC_BATCH = 2

_FRONTEND_PROBE_TIMEOUT_SECONDS = 5


class BookingConsistencyMitigationOracle(Oracle):
    """Diagnostics-aware mitigation oracle for the booking-consistency fault."""

    importance = 1.0

    def __init__(self, problem):
        super().__init__(problem)
        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()

    def evaluate(self, solution=None, trace=None, duration=None) -> dict:
        self._section("Booking Consistency Mitigation Evaluation")

        namespace = self.problem.namespace
        deployment_name = self.problem.frontend_deployment

        self._log(f"namespace: {namespace}")
        self._log(f"deployment under evaluation: {deployment_name}")
        self._log(f"frontend selector: {self.problem.frontend_label_selector}")
        self._log(f"required sidecars: {self.problem.required_sidecars}")
        self._log(f"report path: {self.problem.reports_mount_path}")

        deployment = self._read_deployment(deployment_name, namespace)
        if deployment is None:
            return self._fail("frontend Deployment no longer exists")

        deployment_result = self._check_frontend_deployment_shape(deployment)
        if not deployment_result["success"]:
            return deployment_result

        pressure_result = self._check_report_writer_pressure_is_reduced(deployment)
        if not pressure_result["success"]:
            return pressure_result

        pod_result = self._check_frontend_pods(namespace)
        if not pod_result["success"]:
            return pod_result

        pod_name = pod_result["pod_name"]

        artifact_result = self._check_required_report_artifacts(pod_name, namespace)
        if not artifact_result["success"]:
            return artifact_result

        diagnostics_result = self._check_diagnostics_report_stable(pod_name, namespace)
        if not diagnostics_result["success"]:
            return diagnostics_result

        service_result = self._check_frontend_service_and_endpoints(namespace)
        if not service_result["success"]:
            return service_result

        probe_result = self._check_frontend_http_probe(pod_name, namespace)
        if not probe_result["success"]:
            return probe_result

        self._section("Oracle result")
        self._log("frontend Deployment is present")
        self._log("original frontend container is preserved")
        self._log("required booking-consistency sidecars are present")
        self._log("report-writer diagnostic pressure is structurally reduced")
        self._log("frontend pod is Running and all expected containers are Ready")
        self._log("required report artifacts exist")
        self._log("diagnostics report is healthy in stable samples")
        self._log("frontend Service and Endpoints are present")
        self._log("frontend HTTP probe succeeded")
        self._log("mitigation classification: diagnostics-pressure-reduced")

        return {
            "success": True,
            "details": {
                "pod": pod_name,
                "mitigation_classification": "diagnostics-pressure-reduced",
                "diagnostics_samples": diagnostics_result["samples"],
                "report_writer_env": pressure_result["env"],
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
            self._log(
                f"sidecar {sidecar_name} mounts: "
                f"{[(mount.name, mount.mount_path) for mount in sidecar.volume_mounts or []]}"
            )

            if not self._sidecar_has_required_mounts(sidecar):
                return self._fail(f"sidecar {sidecar_name} does not mount required ledger/report volumes")

        if self.problem.ledger_volume_name not in volume_names:
            return self._fail(f"required volume {self.problem.ledger_volume_name} is missing")

        if self.problem.reports_volume_name not in volume_names:
            return self._fail(f"required volume {self.problem.reports_volume_name} is missing")

        return {"success": True}

    def _check_report_writer_pressure_is_reduced(self, deployment) -> dict:
        self._section("Checking report-writer diagnostic pressure")

        containers = deployment.spec.template.spec.containers or []
        report_writer = self._container_by_name(containers, self.problem.report_writer_name)
        if report_writer is None:
            return self._fail("report-writer sidecar is missing")

        env = {item.name: item.value for item in report_writer.env or []}
        self._log(f"report-writer env: {env}")

        max_workers = self._safe_int(env.get("BOOKING_DIAGNOSTIC_MAX_WORKERS"))
        max_batch = self._safe_int(env.get("BOOKING_DIAGNOSTIC_MAX_BATCH"))
        base_workers = self._safe_int(env.get("BOOKING_DIAGNOSTIC_BASE_WORKERS"))
        base_batch = self._safe_int(env.get("BOOKING_DIAGNOSTIC_BASE_BATCH"))

        if max_workers is None:
            return self._fail("BOOKING_DIAGNOSTIC_MAX_WORKERS is missing or not an integer")

        if max_batch is None:
            return self._fail("BOOKING_DIAGNOSTIC_MAX_BATCH is missing or not an integer")

        if max_workers > _MAX_ACCEPTED_DIAGNOSTIC_WORKERS:
            return self._fail(
                "report-writer still allows too many live validation workers: "
                f"max_workers={max_workers}, threshold={_MAX_ACCEPTED_DIAGNOSTIC_WORKERS}"
            )

        if max_batch > _MAX_ACCEPTED_DIAGNOSTIC_BATCH:
            return self._fail(
                "report-writer still allows too large a live validation batch: "
                f"max_batch={max_batch}, threshold={_MAX_ACCEPTED_DIAGNOSTIC_BATCH}"
            )

        if base_workers is not None and base_workers > _MAX_ACCEPTED_DIAGNOSTIC_WORKERS:
            return self._fail(
                "report-writer base worker count remains too high: "
                f"base_workers={base_workers}, threshold={_MAX_ACCEPTED_DIAGNOSTIC_WORKERS}"
            )

        if base_batch is not None and base_batch > _MAX_ACCEPTED_DIAGNOSTIC_BATCH:
            return self._fail(
                "report-writer base batch remains too high: "
                f"base_batch={base_batch}, threshold={_MAX_ACCEPTED_DIAGNOSTIC_BATCH}"
            )

        return {"success": True, "env": env}

    def _check_frontend_pods(self, namespace: str) -> dict:
        self._section("Checking frontend pods and container readiness")

        pods = self.core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=self.problem.frontend_label_selector,
        ).items

        self._log(f"number of frontend pods found: {len(pods)}")
        if not pods:
            return self._fail("no frontend pods found")

        expected = {
            self.problem.frontend_container_name,
            *self.problem.required_sidecars,
        }

        running_ready_pods = []

        for pod in pods:
            self._log(f"pod: {pod.metadata.name}")
            self._log(f"  phase: {pod.status.phase}")
            self._log(f"  node: {pod.spec.node_name}")
            self._log(f"  pod ip: {pod.status.pod_ip}")

            statuses = pod.status.container_statuses or []
            ready = {status.name for status in statuses if status.ready}
            restarts = {status.name: status.restart_count for status in statuses}

            self._log(f"  ready containers: {sorted(ready)}")
            self._log(f"  restart counts: {restarts}")

            for status in statuses:
                waiting = status.state.waiting
                terminated = status.state.terminated
                if waiting:
                    self._log(f"  container {status.name} waiting: {waiting.reason} {waiting.message}")
                if terminated:
                    self._log(
                        f"  container {status.name} terminated: "
                        f"reason={terminated.reason} exit_code={terminated.exit_code}"
                    )

            if pod.status.phase == "Running" and expected.issubset(ready):
                running_ready_pods.append(pod)

        if not running_ready_pods:
            return self._fail("no frontend pod is Running with all expected containers Ready")

        running_ready_pods.sort(key=lambda pod: pod.status.start_time or 0, reverse=True)
        return {"success": True, "pod_name": running_ready_pods[0].metadata.name}

    def _check_required_report_artifacts(self, pod_name: str, namespace: str) -> dict:
        self._section("Checking required report artifacts")

        report_dir = self.problem.reports_mount_path
        required = [
            "last_successful_run",
            "diagnostics_status.json",
            "report_writer_status.json",
            "booking_consistency_report.json",
        ]

        checks = " && ".join(f"test -s {report_dir}/{name}" for name in required)
        cmd = (
            f"kubectl exec -n {namespace} {pod_name} "
            f"-c {self.problem.report_writer_name} -- "
            f"sh -c '{checks} && echo report_artifacts_ok'"
        )

        output = self.problem.kubectl.exec_command(cmd).strip()
        self._log(f"report artifact check output: {output!r}")

        if "report_artifacts_ok" not in output:
            return self._fail("one or more required report artifacts are missing or empty")

        return {"success": True}

    def _check_diagnostics_report_stable(self, pod_name: str, namespace: str) -> dict:
        self._section("Checking diagnostics report stability")

        samples = []
        healthy_count = 0

        for index in range(_REPORT_SAMPLE_COUNT):
            report = self._read_diagnostics_report(pod_name, namespace)
            samples.append(report)

            self._log(f"sample {index + 1}/{_REPORT_SAMPLE_COUNT}: {report}")

            if self._diagnostics_report_is_healthy(report):
                healthy_count += 1

            if index + 1 < _REPORT_SAMPLE_COUNT:
                time.sleep(_REPORT_SAMPLE_INTERVAL_SECONDS)

        self._log(f"healthy diagnostics samples: {healthy_count}/{_REPORT_SAMPLE_COUNT}")

        if healthy_count < 2:
            return self._fail(
                "diagnostics report did not stay healthy after mitigation: "
                f"healthy_samples={healthy_count}/{_REPORT_SAMPLE_COUNT}"
            )

        return {"success": True, "samples": samples}

    def _read_diagnostics_report(self, pod_name: str, namespace: str) -> dict[str, Any]:
        report_path = f"{self.problem.reports_mount_path}/diagnostics_status.json"
        cmd = (
            f"kubectl exec -n {namespace} {pod_name} "
            f"-c {self.problem.report_writer_name} -- "
            f"cat {report_path}"
        )

        output = self.problem.kubectl.exec_command(cmd).strip()
        if not output:
            raise RuntimeError("diagnostics_status.json is empty")

        # The file should contain one JSON document. If shell output includes
        # extra lines, parse the last JSON-looking line.
        for line in reversed(output.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)

        raise RuntimeError("could not parse diagnostics_status.json output as JSON")

    def _diagnostics_report_is_healthy(self, report: dict[str, Any]) -> bool:
        status = str(report.get("status", ""))
        validation_backend = str(report.get("validation_backend", ""))

        age = self._safe_int(report.get("last_successful_report_age_seconds"))
        p95 = self._safe_int(report.get("p95_duration_ms"))
        p99 = self._safe_int(report.get("p99_duration_ms"))
        active_workers = self._safe_int(report.get("active_workers"))
        max_batch = self._safe_int(report.get("max_batch"))

        try:
            error_rate = float(report.get("error_rate", 1.0))
        except (TypeError, ValueError):
            error_rate = 1.0

        checks = {
            "status_ok_observed": status == "ok",
            "backend_live_mongodb": validation_backend == "live-mongodb",
            "fresh_success": age is not None and 0 <= age <= _MAX_LAST_SUCCESS_AGE_SECONDS,
            "p95_ok": p95 is not None and p95 < _MAX_P95_MS,
            "p99_ok": p99 is not None and p99 < _MAX_P99_MS,
            "error_rate_ok": error_rate < _MAX_ERROR_RATE,
            "active_workers_ok": active_workers is not None and active_workers <= _MAX_ACCEPTED_DIAGNOSTIC_WORKERS,
            "max_batch_ok": max_batch is not None and max_batch <= _MAX_ACCEPTED_DIAGNOSTIC_BATCH,
        }

        hard_checks = {
            key: value
            for key, value in checks.items()
            if key != "status_ok_observed"
        }

        self._log(f"diagnostics health checks: {checks}")
        return all(hard_checks.values())

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

        if selector.get("io.kompose.service") != "frontend":
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

        script = (
            "import urllib.request; "
            f"urllib.request.urlopen('http://127.0.0.1:5000/', timeout={_FRONTEND_PROBE_TIMEOUT_SECONDS}).read(); "
            "print('frontend_probe_ok')"
        )

        cmd = (
            f"kubectl exec -n {namespace} {pod_name} "
            f"-c {self.problem.report_writer_name} -- "
            f"python -c {json.dumps(script)}"
        )

        output = self.problem.kubectl.exec_command(cmd).strip()
        self._log(f"frontend probe output: {output!r}")

        if "frontend_probe_ok" not in output:
            return self._fail("frontend HTTP probe from the pod did not succeed")

        return {"success": True}

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
    def _safe_int(value) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _section(title: str):
        print(f"\n== {title} ==")

    @staticmethod
    def _log(message: str):
        print(f"[booking-consistency-oracle] {message}")

    @classmethod
    def _fail(cls, reason: str) -> dict:
        cls._section("Oracle failure")
        cls._log(f"❌ {reason}")
        return {"success": False, "reason": reason}
