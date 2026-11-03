"""Long-lived Docker container over a SWE-bench instance image (repo at /testbed, conda env testbed)."""
from __future__ import annotations

import io
import re
import shlex
import shutil
import tarfile
import threading
import time
from pathlib import Path, PurePosixPath

import docker
from docker.errors import ImageNotFound, NotFound

from agent.config import CACHE_DIR
from agent.dataset import get_image, get_issue, get_test_cmd
from agent.schemas import ExecResult, TestResult

WORKDIR = "/testbed"
REPRO_FILE = "repro_issue.py"
ENV_PREFIX = "source /opt/miniconda3/bin/activate && conda activate testbed && cd /testbed"
# Django's runner needs a UTF-8 locale (official eval script exports en_US.UTF-8 after locale-gen).
REPO_ENV = {
    "django/django": "export LANG=C.UTF-8 LC_ALL=C.UTF-8",
    # sympy's runner randomizes the hash seed per run → flaky results; pin it for stable regression diffs
    "sympy/sympy": "export PYTHONHASHSEED=0",
}
MAX_OUTPUT_CHARS = 200_000  # keep huge test logs from bloating state.json


def image_name(instance_id: str) -> str:
    """Image per swebench naming (namespace swebench, '__' → '_1776_'); cross-checked with dataset."""
    from swebench.image_builder.image_spec import ImageSpec

    derived = ImageSpec(instance_id=instance_id, dockerfile="-", namespace="swebench").name
    ds_image = get_image(instance_id)
    return ds_image or derived


def ensure_image(client: docker.DockerClient, image: str) -> None:
    try:
        client.images.get(image)
    except ImageNotFound:
        repo, _, tag = image.rpartition(":")
        client.images.pull(repo, tag=tag or "latest")


def _trim(s: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    return s if len(s) <= limit else "...[truncated]...\n" + s[-limit:]


def failure_summary(stdout: str, stderr: str, max_lines: int = 60) -> str:
    """Last ~60 lines of combined output, prioritizing the final traceback."""
    combined = (stdout or "") + ("\n" + stderr if stderr else "")
    lines = combined.rstrip().splitlines()
    tb_idx = max((i for i, ln in enumerate(lines) if ln.startswith("Traceback (most recent call last)")),
                 default=None)
    if tb_idx is not None:
        tb = lines[tb_idx:]
        if len(tb) > max_lines:  # keep head (entry point) + tail (actual error)
            tb = tb[:10] + ["..."] + tb[-(max_lines - 11):]
        return "\n".join(tb)
    return "\n".join(lines[-max_lines:])


def _container_name(run_id: str, instance_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", f"pp.{run_id}.{instance_id}").lower()[:120]


class Sandbox:
    def __init__(self, instance_id: str, run_id: str, *, network: bool = False, name_suffix: str = ""):
        self.instance_id = instance_id
        self.run_id = run_id
        self.repo = get_issue(instance_id).repo
        self.image = image_name(instance_id)
        self.network = network
        self.name = _container_name(run_id, instance_id) + name_suffix
        self.client = docker.from_env(timeout=600)
        self.container = None

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        ensure_image(self.client, self.image)
        try:
            self.client.containers.get(self.name).remove(force=True)
        except NotFound:
            pass
        self.container = self.client.containers.run(
            self.image, command="tail -f /dev/null", name=self.name, detach=True,
            user="root", working_dir=WORKDIR, network_disabled=not self.network,
            mem_limit="6g", labels={"patchpilot": "1", "run_id": self.run_id},
        )
        self.exec("git config --global --add safe.directory /testbed", timeout=30)

    def stop(self) -> None:
        if self.container is not None:
            try:
                self.container.remove(force=True)
            except Exception:
                pass
            self.container = None

    def __enter__(self) -> "Sandbox":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _c(self):
        if self.container is None:
            raise RuntimeError("sandbox not started")
        return self.container

    # ------------------------------------------------------------ exec
    def exec(self, cmd: str, timeout: int = 300, max_output: int = MAX_OUTPUT_CHARS) -> ExecResult:
        env = REPO_ENV.get(self.repo)
        prefix = f"{ENV_PREFIX} && {env}" if env else ENV_PREFIX
        full = f"{prefix} && timeout -k 10 {int(timeout)} bash -c {shlex.quote(cmd)}"
        api = self.client.api
        exec_id = api.exec_create(self._c().id, ["bash", "-lc", full], workdir=WORKDIR, user="root")["Id"]
        box: dict = {}

        def _run():
            try:
                box["out"] = api.exec_start(exec_id, demux=True)
            except Exception as e:  # noqa: BLE001
                box["err"] = e

        t0 = time.time()
        th = threading.Thread(target=_run, daemon=True)
        th.start()
        th.join(timeout + 30)  # inner `timeout` should fire first; this is the safety net
        duration = time.time() - t0
        if th.is_alive():
            try:
                pid = api.exec_inspect(exec_id)["Pid"]
                self._c().exec_run(f"kill -9 {pid}", detach=True)
            except Exception:
                pass
            return ExecResult(exit_code=124, stdout="", stderr=f"host-side timeout after {timeout}s",
                              duration_s=duration, timed_out=True)
        if "err" in box:
            raise box["err"]
        out, err = box.get("out") or (None, None)
        code = api.exec_inspect(exec_id).get("ExitCode")
        code = -1 if code is None else int(code)
        return ExecResult(
            exit_code=code,
            stdout=_trim((out or b"").decode("utf-8", errors="replace"), max_output),
            stderr=_trim((err or b"").decode("utf-8", errors="replace"), max_output),
            duration_s=round(duration, 3), timed_out=code in (124, 137),
        )

    # ------------------------------------------------------------ files
    def _abs(self, path: str) -> PurePosixPath:
        p = PurePosixPath(path)
        return p if p.is_absolute() else PurePosixPath(WORKDIR) / p

    def read_file(self, path: str) -> str:
        p = self._abs(path)
        try:
            stream, _ = self._c().get_archive(str(p))
        except NotFound:
            raise FileNotFoundError(str(p)) from None
        buf = io.BytesIO(b"".join(stream))
        with tarfile.open(fileobj=buf) as tf:
            member = tf.getmembers()[0]
            f = tf.extractfile(member)
            if f is None:
                raise IsADirectoryError(str(p))
            return f.read().decode("utf-8", errors="replace")

    def write_file(self, path: str, content: str) -> None:
        p = self._abs(path)
        # preserve the existing file mode (e.g. 0755 scripts) so diffs don't show spurious mode changes
        r = self._c().exec_run(["bash", "-c", f"mkdir -p {shlex.quote(str(p.parent))} && "
                                              f"(stat -c %a {shlex.quote(str(p))} 2>/dev/null || true)"])
        mode_s = r.output.decode().strip()
        mode = int(mode_s, 8) if re.fullmatch(r"[0-7]{3,4}", mode_s) else 0o644
        data = content.encode("utf-8")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            info = tarfile.TarInfo(name=p.name)
            info.size = len(data)
            info.mode = mode
            info.mtime = int(time.time())
            tf.addfile(info, io.BytesIO(data))
        buf.seek(0)
        if not self._c().put_archive(str(p.parent), buf.getvalue()):
            raise RuntimeError(f"put_archive failed for {p}")

    # ------------------------------------------------------------ git
    def reset(self) -> None:
        r = self.exec("git checkout -- . && git clean -fdq", timeout=120)
        if r.exit_code != 0:
            raise RuntimeError(f"reset failed: {r.stderr[-500:]}")

    def diff(self) -> str:
        # stage everything (so new files appear), diff, then unstage — repro script excluded
        r = self.exec(
            f"git -c core.fileMode=false add -A && "
            f"git -c core.fileMode=false diff --cached -- . ':(exclude){REPRO_FILE}'; "
            f"rc=$?; git reset -q; exit $rc",
            timeout=120,
        )
        if r.exit_code != 0:
            raise RuntimeError(f"diff failed: {r.stderr[-500:]}")
        out = r.stdout
        return out if not out or out.endswith("\n") else out + "\n"

    def export_repo(self, dest: Path | None = None) -> Path:
        """Copy /testbed → cache/repos/<instance_id>/ once (host copy for localization)."""
        dest = Path(dest) if dest else CACHE_DIR / "repos" / self.instance_id
        if (dest / ".pp_export_ok").exists():
            return dest
        tmp = dest.with_name(dest.name + ".partial")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        stream, _ = self._c().get_archive(WORKDIR)
        # stream the tar to disk to avoid holding a large repo in memory
        tar_path = tmp.with_suffix(".tar")
        with open(tar_path, "wb") as f:
            for chunk in stream:
                f.write(chunk)
        with tarfile.open(tar_path) as tf:
            tf.extractall(tmp, filter="tar")
        tar_path.unlink()
        shutil.rmtree(dest, ignore_errors=True)
        (tmp / "testbed").rename(dest)
        shutil.rmtree(tmp, ignore_errors=True)
        (dest / ".pp_export_ok").write_text("ok")
        return dest

    # ------------------------------------------------------------ tests
    def run_repro(self, script: str, timeout: int = 180) -> TestResult:
        self.write_file(REPRO_FILE, script)
        r = self.exec(f"python {REPRO_FILE}", timeout=timeout)
        passed = r.exit_code == 0 and not r.timed_out
        return TestResult(passed=passed, exec=r, failure_summary="" if passed else failure_summary(r.stdout, r.stderr))

    def test_command(self, test_files: list[str]) -> str:
        prefix = get_test_cmd(self.repo)
        if self.repo == "django/django":
            # tests/foo/tests.py → foo.tests (runtests.py takes dotted labels relative to tests/)
            labels = []
            for f in test_files:
                f = f.removeprefix("tests/").removesuffix(".py")
                labels.append(f.replace("/", "."))
            args = labels
        else:
            args = test_files
        return f"{prefix} {' '.join(shlex.quote(a) for a in args)}".strip()

    def run_tests(self, test_files: list[str], timeout: int = 900,
                  max_output: int = MAX_OUTPUT_CHARS) -> TestResult:
        r = self.exec(self.test_command(test_files), timeout=timeout, max_output=max_output)
        passed = r.exit_code == 0 and not r.timed_out
        return TestResult(passed=passed, exec=r, failure_summary="" if passed else failure_summary(r.stdout, r.stderr))
