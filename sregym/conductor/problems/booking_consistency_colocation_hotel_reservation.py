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

    default_interval_seconds = "0"
    default_file_count = "512"
    default_file_mib = "8"
    default_worker_count = "8"
    default_compact_mib = "128"
    default_temp_files = "1"
    default_diagnostic_base_workers = "192"
    default_diagnostic_max_workers = "192"
    default_diagnostic_ramp_scale = "0"
    default_diagnostic_mongo_timeout_ms = "1000"
    default_diagnostic_report_window_seconds = "10"
    default_diagnostic_base_batch = "4"
    default_diagnostic_max_batch = "12"
    default_diagnostic_batch_ramp_seconds = "20"
    default_diagnostic_seed = "13"
    default_diagnostic_mongo_targets = "mongodb-reservation:27017,mongodb-recommendation:27017,mongodb-rate:27017,mongodb-profile:27017"

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
        pod_name = self._wait_for_frontend_pod_ready(timeout=240)

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
                empty_dir=client.V1EmptyDirVolumeSource(size_limit="8Gi"),
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
import hashlib
import json
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

input_path = Path(os.environ["BOOKING_LEDGER_INPUT_PATH"])
archive_path = Path(os.environ["BOOKING_LEDGER_ARCHIVE_PATH"])
report_dir = Path(os.environ["BOOKING_CONSISTENCY_REPORT_DIR"])

interval_seconds = float(os.environ.get("BOOKING_LEDGER_INTERVAL_SECONDS", "0"))
file_count = int(os.environ.get("BOOKING_LEDGER_FILE_COUNT", "512"))
file_mib = int(os.environ.get("BOOKING_LEDGER_FILE_MIB", "8"))
worker_count = int(os.environ.get("BOOKING_LEDGER_WORKER_COUNT", "6"))
compact_mib = int(os.environ.get("BOOKING_LEDGER_COMPACT_MIB", "512"))
temp_files = int(os.environ.get("BOOKING_LEDGER_TEMP_FILES", "8"))

chunk_size = 1024 * 1024
file_bytes = file_mib * chunk_size
compact_bytes = compact_mib * chunk_size

input_path.mkdir(parents=True, exist_ok=True)
archive_path.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)
compact_path = input_path / ".compact"
compact_path.mkdir(parents=True, exist_ok=True)

print("[ledger-reader] starting nuclear bounded ledger compactor", flush=True)
print(f"[ledger-reader] input path: {input_path}", flush=True)
print(f"[ledger-reader] archive path: {archive_path}", flush=True)
print(f"[ledger-reader] report dir: {report_dir}", flush=True)
print(f"[ledger-reader] interval seconds: {interval_seconds}", flush=True)
print(f"[ledger-reader] file count: {file_count}", flush=True)
print(f"[ledger-reader] file mib: {file_mib}", flush=True)
print(f"[ledger-reader] worker count: {worker_count}", flush=True)
print(f"[ledger-reader] compact mib: {compact_mib}", flush=True)
print(f"[ledger-reader] temp files: {temp_files}", flush=True)

def write_file(path: Path, size_bytes: int, seed: bytes):
    block = hashlib.blake2b(seed, digest_size=32).digest()
    remaining = size_bytes
    with path.open("wb") as f:
        while remaining > 0:
            payload = (block * ((min(chunk_size, remaining) // len(block)) + 1))[:min(chunk_size, remaining)]
            f.write(payload)
            remaining -= len(payload)

def initialize_ledgers():
    marker = input_path / ".initialized"
    if marker.exists():
        print("[ledger-reader] ledger artifact set already initialized", flush=True)
        return

    print("[ledger-reader] initializing active ledger artifact set", flush=True)
    for i in range(1, file_count + 1):
        path = input_path / f"ledger-{i}.dat"
        if not path.exists() or path.stat().st_size != file_bytes:
            write_file(path, file_bytes, f"current-{i}".encode())

        # archive exists as a legitimate mitigation target, but keep it smaller
        # so moving the reader there materially reduces the hot working set.
        if i <= max(16, file_count // 16):
            archive = archive_path / f"ledger-{i}.dat"
            if not archive.exists() or archive.stat().st_size != file_bytes:
                write_file(archive, file_bytes, f"archive-{i}".encode())

        if i % 32 == 0:
            print(f"[ledger-reader] initialized {i}/{file_count} active ledger files", flush=True)

    marker.write_text(str(int(time.time())) + "\n")
    print("[ledger-reader] initialization completed", flush=True)

def worker(worker_id: int, status_dir: str):
    status_path = Path(status_dir) / f"ledger_worker_{worker_id}.json"
    rng = random.Random(worker_id + int(time.time()))
    iteration = 0

    while True:
        started = time.time()
        bytes_read = 0
        bytes_written = 0
        digest = hashlib.blake2b(digest_size=32)

        files = list(input_path.glob("ledger-*.dat"))
        rng.shuffle(files)

        # each worker reads a large subset, not necessarily all files.
        target = max(1, len(files) // max(1, worker_count // 2))
        for path in files[:target]:
            try:
                with path.open("rb") as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        digest.update(chunk)
                        bytes_read += len(chunk)
            except FileNotFoundError:
                continue

        compact_id = (iteration * worker_count + worker_id) % max(1, temp_files)
        tmp = compact_path / f"segment-{worker_id}-{compact_id}.tmp"
        final = compact_path / f"segment-{worker_id}-{compact_id}.dat"

        seed = digest.digest()
        block = hashlib.blake2b(seed + str(iteration).encode(), digest_size=64).digest()
        remaining = compact_bytes

        try:
            with tmp.open("wb") as f:
                while remaining > 0:
                    n = min(chunk_size, remaining)
                    payload = (block * ((n // len(block)) + 1))[:n]
                    f.write(payload)
                    bytes_written += len(payload)
                    remaining -= n
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(final)
        except OSError as exc:
            # emptyDir sizeLimit can bite; keep the workload alive and visible.
            print(f"[ledger-reader] worker={worker_id} compaction write failed: {exc}", flush=True)

        duration_ms = int((time.time() - started) * 1000)
        now = int(time.time())
        status = {
            "timestamp": now,
            "worker": worker_id,
            "iteration": iteration,
            "input_path": str(input_path),
            "files_seen": len(files),
            "files_read": target,
            "bytes_read": bytes_read,
            "bytes_written": bytes_written,
            "duration_ms": duration_ms,
            "digest_prefix": digest.hexdigest()[:16],
            "status": "hot",
        }
        status_path.write_text(json.dumps(status, separators=(",", ":")) + "\n")

        print(
            "[ledger-reader] hot worker "
            f"id={worker_id} iter={iteration} files_read={target} "
            f"read={bytes_read} written={bytes_written} duration_ms={duration_ms}",
            flush=True,
        )

        iteration += 1
        if interval_seconds > 0:
            time.sleep(interval_seconds)

def aggregate_status():
    iteration = 0
    while True:
        worker_statuses = []
        for path in report_dir.glob("ledger_worker_*.json"):
            try:
                worker_statuses.append(json.loads(path.read_text()))
            except Exception:
                pass

        now = int(time.time())
        total_read = sum(int(item.get("bytes_read", 0)) for item in worker_statuses)
        total_written = sum(int(item.get("bytes_written", 0)) for item in worker_statuses)
        max_duration = max([int(item.get("duration_ms", 0)) for item in worker_statuses] or [0])

        report = {
            "timestamp": now,
            "input_path": str(input_path),
            "files": len(list(input_path.glob("ledger-*.dat"))),
            "worker_count": worker_count,
            "active_workers": len(worker_statuses),
            "bytes_read_last": total_read,
            "bytes_written_last": total_written,
            "max_worker_duration_ms": max_duration,
            "iteration": iteration,
            "status": "hot",
        }

        (report_dir / "ledger_reader_last_run").write_text(str(now) + "\n")
        (report_dir / "ledger_reader_status.json").write_text(
            json.dumps(report, separators=(",", ":")) + "\n"
        )

        print(
            "[ledger-reader] aggregate "
            f"ts={now} files={report['files']} workers={len(worker_statuses)} "
            f"read_last={total_read} written_last={total_written} "
            f"max_worker_duration_ms={max_duration}",
            flush=True,
        )

        iteration += 1
        time.sleep(5)

initialize_ledgers()

children = []
for worker_id in range(worker_count):
    proc = mp.Process(target=worker, args=(worker_id, str(report_dir)), daemon=False)
    proc.start()
    children.append(proc)

try:
    aggregate_status()
finally:
    for proc in children:
        proc.terminate()
""".strip()

        return client.V1Container(
            name=self.ledger_reader_name,
            image="python:3.12-alpine",
            command=["python", "-u", "-c", script],
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
                requests={"cpu": "100m", "memory": "128Mi"},
                limits={"cpu": "6000m", "memory": "1536Mi"},
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
import json
import math
import os
import queue
import socket
import struct
import threading
import time
from pathlib import Path

report_dir = Path(os.environ["BOOKING_CONSISTENCY_REPORT_DIR"])
report_dir.mkdir(parents=True, exist_ok=True)

initial_success_ts = int(time.time())
(report_dir / "last_successful_run").write_text(str(initial_success_ts) + "\n")
(report_dir / "booking_consistency_report.json").write_text(
    json.dumps({
        "timestamp": initial_success_ts,
        "status": "ok",
        "reason": "initial diagnostics bootstrap",
        "validation_backend": "live-mongodb",
        "last_successful_report_age_seconds": 0,
    }, separators=(",", ":")) + "\n"
)
(report_dir / "diagnostics_status.json").write_text(
    json.dumps({
        "timestamp": initial_success_ts,
        "status": "initializing",
        "reason": "diagnostics workers starting",
        "validation_backend": "live-mongodb",
    }, separators=(",", ":")) + "\n"
)

base_workers = int(os.environ.get("BOOKING_DIAGNOSTIC_BASE_WORKERS", "192"))
max_workers = int(os.environ.get("BOOKING_DIAGNOSTIC_MAX_WORKERS", "192"))
ramp_scale = int(os.environ.get("BOOKING_DIAGNOSTIC_RAMP_SCALE", "0"))
mongo_timeout_ms = int(os.environ.get("BOOKING_DIAGNOSTIC_MONGO_TIMEOUT_MS", "1000"))
report_window_seconds = int(os.environ.get("BOOKING_DIAGNOSTIC_REPORT_WINDOW_SECONDS", "10"))
base_batch = int(os.environ.get("BOOKING_DIAGNOSTIC_BASE_BATCH", "4"))
max_batch = int(os.environ.get("BOOKING_DIAGNOSTIC_MAX_BATCH", "16"))
batch_ramp_seconds = int(os.environ.get("BOOKING_DIAGNOSTIC_BATCH_RAMP_SECONDS", "20"))
diagnostic_seed = int(os.environ.get("BOOKING_DIAGNOSTIC_SEED", "13"))

mongo_targets = [
    target.strip()
    for target in os.environ.get(
        "BOOKING_DIAGNOSTIC_MONGO_TARGETS",
        "mongodb-reservation:27017,mongodb-recommendation:27017,mongodb-rate:27017,mongodb-profile:27017",
    ).split(",")
    if target.strip()
]

print("[booking-diagnostics] starting live Mongo-backed booking diagnostics", flush=True)
print(f"[booking-diagnostics] report dir: {report_dir}", flush=True)
print(f"[booking-diagnostics] mongo targets: {mongo_targets}", flush=True)
print(
    "[booking-diagnostics] worker config "
    f"base={base_workers} max={max_workers} ramp_scale={ramp_scale} "
    f"timeout_ms={mongo_timeout_ms} "
    f"base_batch={base_batch} max_batch={max_batch} batch_ramp_seconds={batch_ramp_seconds}",
    flush=True,
)

started_at = time.time()
events = queue.Queue(maxsize=200000)
scheduler_lock = threading.Lock()


def current_worker_budget(elapsed_seconds: int) -> int:
    ramp = int(math.log2(1 + max(0, elapsed_seconds)) * ramp_scale)
    return max(base_workers, min(max_workers, base_workers + ramp))


def scheduler_snapshot():
    now = time.time()

    with scheduler_lock:
        elapsed = int(now - started_at)
        budget = current_worker_budget(elapsed)
        active_ids = set(range(min(max_workers, budget)))

        return {
            "elapsed_seconds": elapsed,
            "worker_budget": budget,
            "active_workers": len(active_ids),
            "active_worker_ids": active_ids,
        }


def should_worker_run(worker_id: int, snapshot: dict) -> bool:
    return worker_id in snapshot["active_worker_ids"]


def mongo_hello(host: str, port: int, timeout_seconds: float):
    request_id = 1

    body = (
        struct.pack("<i", 16)
        + b"\x10"
        + b"hello\x00"
        + struct.pack("<i", 1)
        + b"\x00"
    )

    flags = 0
    sections = b"\x00" + body
    op_msg = struct.pack("<i", flags) + sections

    message_length = 16 + len(op_msg)
    request = struct.pack("<iiii", message_length, request_id, 0, 2013) + op_msg

    with socket.create_connection((host, port), timeout=timeout_seconds) as sock:
        sock.settimeout(timeout_seconds)
        sock.sendall(request)

        header = sock.recv(16)
        if len(header) != 16:
            raise RuntimeError("short mongo response header")

        response_length, response_request_id, response_to, opcode = struct.unpack("<iiii", header)
        remaining = max(0, response_length - 16)

        received = 0
        while received < remaining:
            chunk = sock.recv(min(4096, remaining - received))
            if not chunk:
                break
            received += len(chunk)

        if response_to != request_id:
            raise RuntimeError(
                f"unexpected mongo response_to={response_to} "
                f"response_request_id={response_request_id} opcode={opcode}"
            )

        return received


def current_batch_size(elapsed_seconds: int) -> int:
    if batch_ramp_seconds <= 0:
        return max_batch

    ramp = elapsed_seconds // batch_ramp_seconds
    return max(base_batch, min(max_batch, base_batch + ramp))


def validate_once(worker_id: int, iteration: int):
    elapsed = int(time.time() - started_at)
    batch_size = current_batch_size(elapsed)

    started = time.time()
    status = "ok"
    error = ""
    bytes_read = 0
    target = ""

    for batch_index in range(batch_size):
        target = mongo_targets[
            (worker_id * 7 + iteration * 3 + batch_index * 5 + diagnostic_seed)
            % len(mongo_targets)
        ]
        host, port_text = target.rsplit(":", 1)
        port = int(port_text)
        timeout_seconds = mongo_timeout_ms / 1000.0

        try:
            bytes_read += mongo_hello(host, port, timeout_seconds)
        except Exception as exc:
            status = "error"
            error = repr(exc)[:200]
            break

    duration_ms = int((time.time() - started) * 1000)

    event = {
        "timestamp": int(time.time()),
        "worker": worker_id,
        "iteration": iteration,
        "target": target,
        "status": status,
        "duration_ms": duration_ms,
        "bytes_read": bytes_read,
        "batch_size": batch_size,
        "error": error,
    }

    try:
        events.put_nowait(event)
    except queue.Full:
        pass

    if iteration % 50 == 0:
        print(
            "[booking-diagnostics] mongo validation "
            f"worker={worker_id} iter={iteration} batch={batch_size} target={target} "
            f"status={status} duration_ms={duration_ms} error={error}",
            flush=True,
        )


def worker(worker_id: int):
    iteration = 0

    while True:
        snapshot = scheduler_snapshot()

        if not should_worker_run(worker_id, snapshot):
            time.sleep(0.25)
            continue

        validate_once(worker_id, iteration)
        iteration += 1


for worker_id in range(max_workers):
    thread = threading.Thread(target=worker, args=(worker_id,), daemon=True)
    thread.start()

window = []
report_iteration = 0

while True:
    deadline = time.time() + 5

    while time.time() < deadline:
        try:
            window.append(events.get(timeout=0.2))
        except queue.Empty:
            pass

    if len(window) > 200000:
        window = window[-200000:]

    now = int(time.time())
    recent = [
        event
        for event in window
        if now - int(event.get("timestamp", 0)) <= report_window_seconds
    ]

    ok = [event for event in recent if event.get("status") == "ok"]
    errors = [event for event in recent if event.get("status") != "ok"]
    durations = sorted(int(event.get("duration_ms", 0)) for event in recent)

    def percentile(values, pct):
        if not values:
            return 0
        index = int((len(values) - 1) * pct)
        return values[index]

    p50_duration_ms = percentile(durations, 0.50)
    p95_duration_ms = percentile(durations, 0.95)
    p99_duration_ms = percentile(durations, 0.99)
    max_duration_ms = max(durations or [0])

    snapshot = scheduler_snapshot()
    current_batch = current_batch_size(snapshot["elapsed_seconds"])
    error_rate = (len(errors) / len(recent)) if recent else 0.0

    previous_success = 0
    last_successful_run_path = report_dir / "last_successful_run"
    if last_successful_run_path.exists():
        try:
            previous_success = int(last_successful_run_path.read_text().strip())
        except Exception:
            previous_success = 0

    degraded = len(recent) > 0 and (
        error_rate >= 0.005
        or len(errors) >= 25
        or p95_duration_ms >= 500
        or p99_duration_ms >= 1000
        or max_duration_ms >= 2500
    )

    status = "degraded" if degraded else "ok"
    if error_rate >= 0.005 or len(errors) >= 25:
        reason = "live mongodb validation timeout"
    elif p95_duration_ms >= 500 or p99_duration_ms >= 1000 or max_duration_ms >= 2500:
        reason = "live mongodb validation tail latency"
    else:
        reason = "live mongodb validation healthy"

    if not degraded:
        last_successful_run_path.write_text(str(now) + "\n")
        previous_success = now

    last_successful_age = now - previous_success if previous_success else -1

    report = {
        "timestamp": now,
        "status": status,
        "reason": reason,
        "validation_backend": "live-mongodb",
        "mongo_targets": mongo_targets,
        "base_workers": base_workers,
        "max_workers": max_workers,
        "worker_budget": snapshot["worker_budget"],
        "active_workers": snapshot["active_workers"],
        "report_window_seconds": report_window_seconds,
        "current_batch_size": current_batch,
        "base_batch": base_batch,
        "max_batch": max_batch,
        "batch_ramp_seconds": batch_ramp_seconds,
        "requests_last_window": len(recent),
        "ok_last_window": len(ok),
        "errors_last_window": len(errors),
        "error_rate": round(error_rate, 4),
        "p50_duration_ms": p50_duration_ms,
        "p95_duration_ms": p95_duration_ms,
        "p99_duration_ms": p99_duration_ms,
        "max_duration_ms": max_duration_ms,
        "last_successful_report_age_seconds": last_successful_age,
        "iteration": report_iteration,
    }

    (report_dir / "diagnostics_status.json").write_text(
        json.dumps(report, separators=(",", ":")) + "\n"
    )
    (report_dir / "report_writer_status.json").write_text(
        json.dumps(report, separators=(",", ":")) + "\n"
    )
    (report_dir / "booking_consistency_report.json").write_text(
        json.dumps(report, separators=(",", ":")) + "\n"
    )

    print(
        "[booking-diagnostics] wrote report "
        f"status={status} requests={len(recent)} errors={len(errors)} "
        f"error_rate={error_rate:.3f} budget={snapshot['worker_budget']} "
        f"active={snapshot['active_workers']} batch={current_batch} "
        f"p95_ms={p95_duration_ms} p99_ms={p99_duration_ms} "
        f"max_ms={max_duration_ms} last_success_age={last_successful_age}",
        flush=True,
    )

    report_iteration += 1
""".strip()

        return client.V1Container(
            name=self.report_writer_name,
            image="python:3.12-alpine",
            command=["python", "-u", "-c", script],
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
                requests={"cpu": "100m", "memory": "128Mi"},
                limits={"cpu": "3000m", "memory": "768Mi"},
            ),
        )

    def _common_env(self):
        return [
            client.V1EnvVar(name=self.input_env, value=self.original_input_path),
            client.V1EnvVar(name=self.archive_env, value=self.alternate_input_path),
            client.V1EnvVar(name=self.interval_env, value=self.default_interval_seconds),
            client.V1EnvVar(name=self.file_count_env, value=self.default_file_count),
            client.V1EnvVar(name=self.file_mib_env, value=self.default_file_mib),
            client.V1EnvVar(name="BOOKING_LEDGER_WORKER_COUNT", value=self.default_worker_count),
            client.V1EnvVar(name="BOOKING_LEDGER_COMPACT_MIB", value=self.default_compact_mib),
            client.V1EnvVar(name="BOOKING_LEDGER_TEMP_FILES", value=self.default_temp_files),
            client.V1EnvVar(name=self.reports_env, value=self.reports_mount_path),
            client.V1EnvVar(name="BOOKING_DIAGNOSTIC_BASE_WORKERS", value=self.default_diagnostic_base_workers),
            client.V1EnvVar(name="BOOKING_DIAGNOSTIC_MAX_WORKERS", value=self.default_diagnostic_max_workers),
            client.V1EnvVar(name="BOOKING_DIAGNOSTIC_RAMP_SCALE", value=self.default_diagnostic_ramp_scale),
            client.V1EnvVar(name="BOOKING_DIAGNOSTIC_MONGO_TIMEOUT_MS", value=self.default_diagnostic_mongo_timeout_ms),
            client.V1EnvVar(name="BOOKING_DIAGNOSTIC_REPORT_WINDOW_SECONDS", value=self.default_diagnostic_report_window_seconds),
            client.V1EnvVar(name="BOOKING_DIAGNOSTIC_BASE_BATCH", value=self.default_diagnostic_base_batch),
            client.V1EnvVar(name="BOOKING_DIAGNOSTIC_MAX_BATCH", value=self.default_diagnostic_max_batch),
            client.V1EnvVar(name="BOOKING_DIAGNOSTIC_BATCH_RAMP_SECONDS", value=self.default_diagnostic_batch_ramp_seconds),
            client.V1EnvVar(name="BOOKING_DIAGNOSTIC_SEED", value=self.default_diagnostic_seed),
            client.V1EnvVar(name="BOOKING_DIAGNOSTIC_MONGO_TARGETS", value=self.default_diagnostic_mongo_targets),
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
                    return pod.metadata.name

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

        expected = {
            self.frontend_container_name,
            self.ledger_reader_name,
            self.consistency_checker_name,
            self.report_writer_name,
        }

        running = [pod for pod in pods if pod.status.phase == "Running"]
        preferred = []

        for pod in running:
            spec_names = {container.name for container in pod.spec.containers or []}
            ready_names = {
                status.name
                for status in (pod.status.container_statuses or [])
                if status.ready
            }

            if expected.issubset(spec_names) and expected.issubset(ready_names):
                preferred.append(pod)

        if preferred:
            preferred.sort(key=lambda pod: pod.status.start_time or 0, reverse=True)
            return preferred[0].metadata.name

        if running:
            running.sort(key=lambda pod: pod.status.start_time or 0, reverse=True)
            return running[0].metadata.name

        return None

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

