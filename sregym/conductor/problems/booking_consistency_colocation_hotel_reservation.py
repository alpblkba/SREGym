"""Booking consistency background work on Hotel Reservation.

this problem injects a required background consistency workload into the
Hotel Reservation frontend pod. the frontend application container remains
unchanged and healthy, but the pod gains additional containers that maintain
reservation-ledger artifacts and write periodic consistency reports.

the intended incident shape is deliberately not a crash, a service selector
bug, a database outage, or an obvious CPU saturation case. the application
continues to serve traffic, but request timing becomes worse while the
background ledger work is active.

a correct mitigation must preserve the background consistency workload and
its reports. removing the sidecars, disabling the report writer, deleting the
frontend deployment, or simply restarting the pod is not a valid fix.
"""

import copy
import time

from kubernetes import client

from sregym.conductor.oracles.booking_consistency_mitigation_oracle import (
    BookingConsistencyMitigationOracle,
)
from sregym.conductor.oracles.llm_as_a_judge.llm_as_a_judge_oracle import LLMAsAJudgeOracle
from sregym.conductor.problems.base import Problem
from sregym.service.apps.hotel_reservation import HotelReservation
from sregym.service.kubectl import KubeCtl
from sregym.utils.decorators import mark_fault_injected


class BookingConsistencyColocationHotelReservation(Problem):
    """Inject a required background consistency workload into the frontend pod."""

    frontend_deployment = "frontend"
    frontend_label_selector = "io.kompose.service=frontend"
    frontend_container_name = "hotel-reserv-frontend"
    frontend_image = "yinfangchen/hotelreservation:latest"
    frontend_command = ["frontend"]

    ledger_volume_name = "reservation-ledger"
    reports_volume_name = "consistency-reports"

    ledger_reader_name = "ledger-reader"
    consistency_checker_name = "consistency-checker"
    report_writer_name = "report-writer"

    ledger_mount_path = "/var/lib/booking-ledger"
    reports_mount_path = "/var/lib/booking-consistency/reports"

    original_input_path = "/var/lib/booking-ledger/current"
    alternate_input_path = "/var/lib/booking-ledger/archive"

    input_env = "BOOKING_LEDGER_INPUT_PATH"
    archive_env = "BOOKING_LEDGER_ARCHIVE_PATH"
    interval_env = "BOOKING_LEDGER_INTERVAL_SECONDS"
    file_count_env = "BOOKING_LEDGER_FILE_COUNT"
    file_mib_env = "BOOKING_LEDGER_FILE_MIB"
    reports_env = "BOOKING_CONSISTENCY_REPORT_DIR"

    default_interval_seconds = "1"
    default_file_count = "96"
    default_file_mib = "2"

    required_sidecars = [
        ledger_reader_name,
        consistency_checker_name,
        report_writer_name,
    ]

    def __init__(self):
        super().__init__(app=HotelReservation())

        self.kubectl = KubeCtl()
        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
        self._original_frontend_template = None

        self.root_cause = self.build_structured_root_cause(
            component=f"deployment/{self.frontend_deployment}",
            namespace=self.namespace,
            description=(
                "The frontend pod contains required booking-consistency sidecars. "
                "The ledger-reader sidecar repeatedly processes the active reservation "
                "ledger artifact set at a high frequency. The frontend application "
                "container itself is not misconfigured, the service selector is still "
                "correct, and the backing services remain healthy. A valid mitigation "
                "must preserve the consistency workload and its reports while changing "
                "the background ledger work so it no longer degrades frontend request "
                "timing."
            ),
        )
        self.diagnosis_oracle = LLMAsAJudgeOracle(problem=self, expected=self.root_cause)
        self.mitigation_oracle = BookingConsistencyMitigationOracle(problem=self)

        self.app.create_workload()

    @mark_fault_injected
    def inject_fault(self):
        self._log_step("== Fault Injection ==")
        self._log_step(f"namespace: {self.namespace}")
        self._log_step(f"target deployment: {self.frontend_deployment}")
        self._log_step(f"target selector: {self.frontend_label_selector}")
        self._log_step("reading current frontend Deployment")

        deployment = self.apps_v1.read_namespaced_deployment(
            name=self.frontend_deployment,
            namespace=self.namespace,
        )
        self._original_frontend_template = copy.deepcopy(deployment.spec.template)

        original_containers = [c.name for c in deployment.spec.template.spec.containers or []]
        original_volumes = [v.name for v in deployment.spec.template.spec.volumes or []]

        self._log_step(f"original containers: {original_containers}")
        self._log_step(f"original volumes: {original_volumes}")
        self._log_step("building replacement frontend pod template")
        self._log_step(f"injecting sidecars: {self.required_sidecars}")
        self._log_step(
            "ledger workload config: "
            f"input={self.original_input_path}, "
            f"archive={self.alternate_input_path}, "
            f"interval={self.default_interval_seconds}s, "
            f"files={self.default_file_count}, "
            f"file_mib={self.default_file_mib}"
        )

        updated_deployment = self._deployment_with_booking_consistency_workload(deployment)

        self._log_step("replacing frontend Deployment with updated pod template")
        self.apps_v1.replace_namespaced_deployment(
            name=self.frontend_deployment,
            namespace=self.namespace,
            body=updated_deployment,
        )

        self._log_step("waiting for frontend rollout to finish")
        rollout_output = self.kubectl.exec_command(
            f"kubectl rollout status deployment/{self.frontend_deployment} "
            f"-n {self.namespace} --timeout=180s"
        )
        self._log_step(f"rollout output: {rollout_output.strip()}")

        self._log_step("waiting for frontend pod and all booking-consistency containers to become Ready")
        self._wait_for_frontend_pod_ready(timeout=180)

        pod_name = self._frontend_pod_name()
        self._log_step(f"current frontend pod: {pod_name}")

        self._log_step("waiting for report artifacts from report-writer")
        self._wait_for_report_artifacts(timeout=120)

        self._log_step("report artifacts exist")
        self._log_step("fault injection completed successfully")

    @mark_fault_injected
    def recover_fault(self):
        self._log_step("== Fault Recovery ==")
        self._log_step(f"namespace: {self.namespace}")
        self._log_step(f"target deployment: {self.frontend_deployment}")

        deployment = self.apps_v1.read_namespaced_deployment(
            name=self.frontend_deployment,
            namespace=self.namespace,
        )

        # prefer restoring the exact template captured during injection when the
        # same Problem instance is still alive. if not available, perform a
        # conservative cleanup that removes only the injected containers and
        # volumes from the current deployment object.
        if self._original_frontend_template is not None:
            self._log_step("restoring original frontend pod template captured during injection")
            deployment.spec.template = copy.deepcopy(self._original_frontend_template)
        else:
            self._log_step("original template snapshot unavailable; removing injected sidecars and volumes")
            self._remove_injected_sidecars_and_volumes_from_deployment(deployment)

        self.apps_v1.replace_namespaced_deployment(
            name=self.frontend_deployment,
            namespace=self.namespace,
            body=deployment,
        )

        self._log_step("waiting for frontend rollout after recovery")
        rollout_output = self.kubectl.exec_command(
            f"kubectl rollout status deployment/{self.frontend_deployment} "
            f"-n {self.namespace} --timeout=180s"
        )
        self._log_step(f"recovery rollout output: {rollout_output.strip()}")
        self._log_step("fault recovery completed")

    def _deployment_with_booking_consistency_workload(self, deployment):
        template = deployment.spec.template
        pod_spec = template.spec

        existing_containers = [
            c for c in (pod_spec.containers or []) if c.name not in self.required_sidecars
        ]
        existing_volumes = [
            v
            for v in (pod_spec.volumes or [])
            if v.name not in (self.ledger_volume_name, self.reports_volume_name)
        ]

        containers = existing_containers + [
            self._ledger_reader_container(),
            self._consistency_checker_container(),
            self._report_writer_container(),
        ]

        volumes = existing_volumes + [
            client.V1Volume(
                name=self.ledger_volume_name,
                empty_dir=client.V1EmptyDirVolumeSource(),
            ),
            client.V1Volume(
                name=self.reports_volume_name,
                empty_dir=client.V1EmptyDirVolumeSource(),
            ),
        ]

        annotations = dict(template.metadata.annotations or {})
        annotations["ops.hotelreservation.io/consistency-workload"] = "reservation-ledger"

        labels = dict(template.metadata.labels or {})
        labels["ops.hotelreservation.io/booking-consistency"] = "enabled"

        self._log_step(f"replacement containers: {[c.name for c in containers]}")
        self._log_step(f"replacement volumes: {[v.name for v in volumes]}")
        self._log_step(f"replacement labels include: {labels}")
        self._log_step(f"replacement annotations include: {annotations}")

        template.metadata.annotations = annotations
        template.metadata.labels = labels
        pod_spec.containers = containers
        pod_spec.volumes = volumes

        return deployment

    def _ledger_reader_container(self):
        script = r"""
set -eu

echo "[ledger-reader] starting"
echo "[ledger-reader] input path: $BOOKING_LEDGER_INPUT_PATH"
echo "[ledger-reader] archive path: $BOOKING_LEDGER_ARCHIVE_PATH"
echo "[ledger-reader] report dir: $BOOKING_CONSISTENCY_REPORT_DIR"
echo "[ledger-reader] interval seconds: $BOOKING_LEDGER_INTERVAL_SECONDS"
echo "[ledger-reader] file count: $BOOKING_LEDGER_FILE_COUNT"
echo "[ledger-reader] file mib: $BOOKING_LEDGER_FILE_MIB"

mkdir -p "$BOOKING_LEDGER_INPUT_PATH" "$BOOKING_LEDGER_ARCHIVE_PATH" "$BOOKING_CONSISTENCY_REPORT_DIR"

if [ ! -f "$BOOKING_LEDGER_INPUT_PATH/.initialized" ]; then
  echo "[ledger-reader] initializing ledger artifact set"
  i=1
  while [ "$i" -le "$BOOKING_LEDGER_FILE_COUNT" ]; do
    dd if=/dev/zero of="$BOOKING_LEDGER_INPUT_PATH/ledger-$i.dat" bs=1M count="$BOOKING_LEDGER_FILE_MIB" 2>/dev/null
    dd if=/dev/zero of="$BOOKING_LEDGER_ARCHIVE_PATH/ledger-$i.dat" bs=1M count="$BOOKING_LEDGER_FILE_MIB" 2>/dev/null
    if [ $((i % 16)) -eq 0 ]; then
      echo "[ledger-reader] initialized $i ledger artifacts"
    fi
    i=$((i + 1))
  done
  date +%s > "$BOOKING_LEDGER_INPUT_PATH/.initialized"
  echo "[ledger-reader] initialization completed"
else
  echo "[ledger-reader] ledger artifact set already initialized"
fi

while true; do
  start=$(date +%s)
  files=0

  for file in "$BOOKING_LEDGER_INPUT_PATH"/ledger-*.dat; do
    if [ -f "$file" ]; then
      files=$((files + 1))
      cat "$file" >/dev/null
    fi
  done

  end=$(date +%s)
  duration=$((end - start))
  printf '%s\n' "$end" > "$BOOKING_CONSISTENCY_REPORT_DIR/ledger_reader_last_run"
  printf '{"timestamp":%s,"input_path":"%s","files":%s,"duration_seconds":%s}\n' \
    "$end" "$BOOKING_LEDGER_INPUT_PATH" "$files" "$duration" \
    > "$BOOKING_CONSISTENCY_REPORT_DIR/ledger_reader_status.json"
  echo "[ledger-reader] scan done: ts=$end path=$BOOKING_LEDGER_INPUT_PATH files=$files duration=${duration}s"
  sleep "$BOOKING_LEDGER_INTERVAL_SECONDS"
done
""".strip()

        return client.V1Container(
            name=self.ledger_reader_name,
            image="busybox:1.36",
            command=["sh", "-c", script],
            env=self._common_env(),
            volume_mounts=[
                client.V1VolumeMount(
                    name=self.ledger_volume_name,
                    mount_path=self.ledger_mount_path,
                ),
                client.V1VolumeMount(
                    name=self.reports_volume_name,
                    mount_path=self.reports_mount_path,
                ),
            ],
            resources=client.V1ResourceRequirements(
                requests={"cpu": "10m", "memory": "32Mi"},
                limits={"cpu": "200m", "memory": "512Mi"},
            ),
        )

    def _consistency_checker_container(self):
        script = r"""
set -eu

echo "[consistency-checker] starting"
echo "[consistency-checker] input path: $BOOKING_LEDGER_INPUT_PATH"
echo "[consistency-checker] report dir: $BOOKING_CONSISTENCY_REPORT_DIR"

mkdir -p "$BOOKING_CONSISTENCY_REPORT_DIR"

while true; do
  count=$(find "$BOOKING_LEDGER_INPUT_PATH" -type f -name 'ledger-*.dat' | wc -l)
  now=$(date +%s)
  printf '{"timestamp":%s,"input_path":"%s","ledger_files":%s,"status":"checked"}\n' \
    "$now" "$BOOKING_LEDGER_INPUT_PATH" "$count" \
    > "$BOOKING_CONSISTENCY_REPORT_DIR/checker_status.json"
  echo "[consistency-checker] checked ledger files: ts=$now path=$BOOKING_LEDGER_INPUT_PATH count=$count"
  sleep 5
done
""".strip()

        return client.V1Container(
            name=self.consistency_checker_name,
            image="busybox:1.36",
            command=["sh", "-c", script],
            env=self._common_env(),
            volume_mounts=[
                client.V1VolumeMount(
                    name=self.ledger_volume_name,
                    mount_path=self.ledger_mount_path,
                ),
                client.V1VolumeMount(
                    name=self.reports_volume_name,
                    mount_path=self.reports_mount_path,
                ),
            ],
            resources=client.V1ResourceRequirements(
                requests={"cpu": "5m", "memory": "16Mi"},
                limits={"cpu": "100m", "memory": "128Mi"},
            ),
        )

    def _report_writer_container(self):
        script = r"""
set -eu

echo "[report-writer] starting"
echo "[report-writer] report dir: $BOOKING_CONSISTENCY_REPORT_DIR"

mkdir -p "$BOOKING_CONSISTENCY_REPORT_DIR"

while true; do
  now=$(date +%s)
  reader_last_run="unknown"
  if [ -f "$BOOKING_CONSISTENCY_REPORT_DIR/ledger_reader_last_run" ]; then
    reader_last_run=$(cat "$BOOKING_CONSISTENCY_REPORT_DIR/ledger_reader_last_run")
  fi

  printf '%s\n' "$now" > "$BOOKING_CONSISTENCY_REPORT_DIR/last_successful_run"

  printf '{"timestamp":%s,"reader_last_run":"%s","status":"ok"}\n' \
    "$now" "$reader_last_run" \
    > "$BOOKING_CONSISTENCY_REPORT_DIR/booking_consistency_report.json"

  echo "[report-writer] wrote report: ts=$now reader_last_run=$reader_last_run"
  sleep 5
done
""".strip()

        return client.V1Container(
            name=self.report_writer_name,
            image="busybox:1.36",
            command=["sh", "-c", script],
            env=self._common_env(),
            volume_mounts=[
                client.V1VolumeMount(
                    name=self.ledger_volume_name,
                    mount_path=self.ledger_mount_path,
                ),
                client.V1VolumeMount(
                    name=self.reports_volume_name,
                    mount_path=self.reports_mount_path,
                ),
            ],
            resources=client.V1ResourceRequirements(
                requests={"cpu": "5m", "memory": "16Mi"},
                limits={"cpu": "100m", "memory": "128Mi"},
            ),
        )

    def _common_env(self):
        return [
            client.V1EnvVar(name=self.input_env, value=self.original_input_path),
            client.V1EnvVar(name=self.archive_env, value=self.alternate_input_path),
            client.V1EnvVar(name=self.interval_env, value=self.default_interval_seconds),
            client.V1EnvVar(name=self.file_count_env, value=self.default_file_count),
            client.V1EnvVar(name=self.file_mib_env, value=self.default_file_mib),
            client.V1EnvVar(name=self.reports_env, value=self.reports_mount_path),
        ]

    def _wait_for_frontend_pod_ready(self, timeout: int = 180):
        deadline = time.monotonic() + timeout
        last_seen = None

        while time.monotonic() < deadline:
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=self.frontend_label_selector,
            ).items

            status_lines = []
            for pod in pods:
                statuses = pod.status.container_statuses or []
                ready = {cs.name for cs in statuses if cs.ready}
                waiting = [
                    f"{cs.name}:{cs.state.waiting.reason}"
                    for cs in statuses
                    if cs.state and cs.state.waiting and cs.state.waiting.reason
                ]
                restarts = {cs.name: cs.restart_count for cs in statuses}
                status_lines.append(
                    f"pod={pod.metadata.name} phase={pod.status.phase} "
                    f"ready={sorted(ready)} waiting={waiting} restarts={restarts}"
                )

                expected = {
                    self.frontend_container_name,
                    self.ledger_reader_name,
                    self.consistency_checker_name,
                    self.report_writer_name,
                }

                if pod.status.phase == "Running" and expected.issubset(ready):
                    self._log_step(f"frontend pod is Ready with expected containers: {pod.metadata.name}")
                    return

            joined = " | ".join(status_lines) if status_lines else "no frontend pods found"
            if joined != last_seen:
                self._log_step(f"readiness wait: {joined}")
                last_seen = joined

            time.sleep(3)

        raise RuntimeError("frontend pod with booking-consistency sidecars did not become Ready")

    def _wait_for_report_artifacts(self, timeout: int = 120):
        deadline = time.monotonic() + timeout
        last_error = None

        while time.monotonic() < deadline:
            pod_name = self._frontend_pod_name()
            if pod_name:
                self._log_step(f"checking report artifacts in pod={pod_name}")
                cmd = (
                    f"kubectl exec -n {self.namespace} {pod_name} "
                    f"-c {self.report_writer_name} -- "
                    f"sh -c 'test -s {self.reports_mount_path}/last_successful_run "
                    f"&& test -s {self.reports_mount_path}/booking_consistency_report.json "
                    f"&& echo ok'"
                )
                try:
                    output = self.kubectl.exec_command(cmd).strip()
                    self._log_step(f"report artifact check output: {output!r}")
                    if "ok" in output.lower():
                        return
                except Exception as exc:
                    error = repr(exc)
                    if error != last_error:
                        self._log_step(f"report artifact check failed: {error}")
                        last_error = error
            else:
                self._log_step("report artifact check skipped: no running frontend pod yet")
            time.sleep(3)

        raise RuntimeError("booking consistency report artifacts were not created")

    def _frontend_pod_name(self) -> str | None:
        pods = self.core_v1.list_namespaced_pod(
            namespace=self.namespace,
            label_selector=self.frontend_label_selector,
        ).items
        running = [pod for pod in pods if pod.status.phase == "Running"]
        if not running:
            return None
        return running[0].metadata.name

    def _remove_injected_sidecars_and_volumes_from_deployment(self, deployment):
        pod_spec = deployment.spec.template.spec

        containers = [
            c
            for c in (pod_spec.containers or [])
            if c.name not in self.required_sidecars
        ]
        volumes = [
            v
            for v in (pod_spec.volumes or [])
            if v.name not in (self.ledger_volume_name, self.reports_volume_name)
        ]

        self._log_step(f"cleanup containers after removal: {[c.name for c in containers]}")
        self._log_step(f"cleanup volumes after removal: {[v.name for v in volumes]}")

        pod_spec.containers = containers
        pod_spec.volumes = volumes

    @staticmethod
    def _log_step(message: str):
        print(f"[booking-consistency] {message}")

