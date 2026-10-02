# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime inventory records evidence without treating declarations as proof."""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector.sglang_rubin import runtime

pytestmark = pytest.mark.unit


def _runtime_subprocess(tmp_path, script):
    package_root = Path(runtime.__file__).resolve().parents[2]
    code = (
        "import os, sys\n"
        f"sys.path[:0] = [{str(tmp_path)!r}, {str(package_root)!r}]\n"
        "from collector.sglang_rubin import runtime\n" + textwrap.dedent(script)
    )
    return subprocess.run([sys.executable, "-I", "-S", "-c", code], capture_output=True, text=True, check=False)


@pytest.fixture
def rubin_runtime(monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "aarch64")
    for name, value in runtime.EXPECTED_BUILD_ENV.items():
        monkeypatch.setenv(name, value)
    for name, value in runtime.REQUIRED_SERVING_ENV.items():
        monkeypatch.setenv(name, value)
    # Distribution metadata may differ from the image's SGLANG_VERSION label.
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda name: "0.5.18.post1")
    device = SimpleNamespace(name="NVIDIA Graphics Device", total_memory=288 * 1024**3, uuid="GPU-test")
    torch = SimpleNamespace(
        __version__="2.12.0a0",
        version=SimpleNamespace(cuda="13.5"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda index: device,
            get_device_capability=lambda index: (10, 7),
        ),
    )
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: torch)
    return torch


def test_cpu_import_does_not_load_frameworks():
    package_root = Path(runtime.__file__).resolve().parents[2]
    code = (
        f"import sys; sys.path.insert(0, {str(package_root)!r}); "
        "import collector.sglang_rubin.runtime; "
        "assert not {'torch', 'sglang', 'aisimulate'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-I", "-S", "-c", code], check=True)


def test_observes_devices_and_separates_build_claims(rubin_runtime, monkeypatch):
    monkeypatch.setenv("GITLAB_TOKEN", "must-not-appear")
    inventory = runtime.collect_inventory(launcher_image=runtime.IMAGE_REF)
    assert runtime.validate_runtime(inventory) == []
    assert inventory["launcher_provenance"] == {"image": runtime.IMAGE_REF, "image_identity_verified": False}
    assert inventory["declared_image"]["sglang_distribution_version"] is None
    assert inventory["declared_serving_configuration"] == {
        "environment": {"SGLANG_ENABLE_MOE_DEFERRED_FINALIZE": "0"},
        "server_args": {"disable_prefill_cuda_graph": True},
    }
    assert inventory["observed"]["serving_environment"] == {"SGLANG_ENABLE_MOE_DEFERRED_FINALIZE": "0"}
    assert inventory["observed"]["package_versions"]["sglang"]["version"] == "0.5.18.post1"
    cuda = inventory["observed"]["cuda"]
    assert cuda["torch_cuda_version"] == "13.5"
    assert cuda["devices"] == [
        {
            "index": 0,
            "name": "NVIDIA Graphics Device",
            "capability": [10, 7],
            "total_memory_bytes": 288 * 1024**3,
            "uuid": "GPU-test",
        }
    ]
    assert "GITLAB_TOKEN" not in json.dumps(inventory)
    assert "must-not-appear" not in json.dumps(inventory)


def test_declared_digest_does_not_override_incompatible_device(rubin_runtime):
    rubin_runtime.cuda.get_device_capability = lambda index: (10, 0)
    inventory = runtime.collect_inventory(launcher_image=runtime.IMAGE_REF)
    assert any("requires SM107" in error for error in runtime.validate_runtime(inventory))


def test_missing_observations_fail_validation():
    errors = runtime.validate_runtime({"declared_image": {"reference": runtime.IMAGE_REF}})
    assert any("Linux/aarch64" in error for error in errors)
    assert any("SGLANG_VERSION" in error for error in errors)
    assert any("visible CUDA device" in error for error in errors)


@pytest.mark.parametrize("name", runtime.EXPECTED_BUILD_ENV)
def test_build_metadata_must_be_present_and_match(rubin_runtime, monkeypatch, name):
    monkeypatch.delenv(name)
    assert any(name in error for error in runtime.validate_runtime(runtime.collect_inventory()))
    monkeypatch.setenv(name, "a-different-build")
    assert any(name in error for error in runtime.validate_runtime(runtime.collect_inventory()))


@pytest.mark.parametrize("value", [None, "1", "false"])
def test_serving_environment_must_be_explicit_and_is_never_overridden(rubin_runtime, monkeypatch, value):
    name = "SGLANG_ENABLE_MOE_DEFERRED_FINALIZE"
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    inventory = runtime.collect_inventory()

    assert inventory["observed"]["serving_environment"][name] == value
    assert any(f"Serving environment {name}" in error for error in runtime.validate_runtime(inventory))
    assert os.environ.get(name) == value


def test_cuda_probe_failure_retains_error(rubin_runtime):
    def fail():
        raise RuntimeError("CUDA initialization failed")

    rubin_runtime.cuda.is_available = fail
    inventory = runtime.collect_inventory()
    assert inventory["observed"]["cuda"]["error"] == "RuntimeError: CUDA initialization failed"
    assert any("CUDA initialization failed" in error for error in runtime.validate_runtime(inventory))


def test_missing_package_metadata_is_explicit(rubin_runtime, monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(runtime.importlib.metadata, "version", missing)
    inventory = runtime.collect_inventory()
    assert inventory["observed"]["package_versions"]["nixl"]["version"] is None
    assert "PackageNotFoundError" in inventory["observed"]["package_versions"]["nixl"]["error"]
    assert any("sglang" in error for error in runtime.validate_runtime(inventory))


@pytest.mark.parametrize("name", ["sglang", "nixl"])
def test_cli_preserves_invalid_distribution_metadata_error(tmp_path, name):
    metadata_dir = tmp_path / f"{name}-0.5.18.dist-info"
    metadata_dir.mkdir()
    (metadata_dir / "METADATA").write_bytes(f"Name: {name}\nVersion: 0.5.18\n".encode() + b"\xff")
    result = _runtime_subprocess(tmp_path, 'raise SystemExit(runtime.main(["--validate"]))')

    assert result.returncode == 1
    inventory = json.loads(result.stdout)
    package = inventory["observed"]["package_versions"][name]
    assert package["version"] is None
    assert package["error_type"] == "UnicodeDecodeError"
    assert "UnicodeDecodeError" in package["error"]
    assert any(name in error and "UnicodeDecodeError" in error for error in inventory["validation"]["errors"])


def test_missing_optional_distribution_does_not_fail_validation(rubin_runtime, monkeypatch):
    def version(name):
        if name == "flashinfer":
            raise importlib.metadata.PackageNotFoundError(name)
        return "0.5.18.post1"

    monkeypatch.setattr(runtime.importlib.metadata, "version", version)
    assert runtime.validate_runtime(runtime.collect_inventory()) == []


def test_optional_native_import_failure_fails_preflight(rubin_runtime, monkeypatch):
    def import_module(name):
        if name == "torch":
            return rubin_runtime
        if name == "aisimulate._runtime":
            raise ImportError("undefined symbol")
        return SimpleNamespace(__file__=f"/site-packages/{name}/__init__.py")

    monkeypatch.setattr(runtime.importlib, "import_module", import_module)
    inventory = runtime.collect_inventory(check_imports=True)
    assert inventory["observed"]["imports"]["aisimulate._runtime"]["error"] == "ImportError: undefined symbol"
    assert any("aisimulate._runtime" in error for error in runtime.validate_runtime(inventory))


def test_checkpoint_hashes_only_metadata_and_reports_missing_config(rubin_runtime, tmp_path):
    payload = b'{"quantization": {"quant_algo": "NVFP4"}}'
    (tmp_path / "hf_quant_config.json").write_bytes(payload)
    (tmp_path / "credentials.env").write_text("PRIVATE_TOKEN=secret")
    inventory = runtime.collect_inventory(checkpoint_dir=tmp_path)
    files = inventory["observed"]["checkpoint"]["files"]
    assert files["hf_quant_config.json"]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert files["quantization_config.json"] == {"present": False}
    assert "credentials.env" not in files
    assert any("config.json" in error for error in runtime.validate_runtime(inventory))
    (tmp_path / "config.json").write_text("{}")
    assert runtime.validate_runtime(runtime.collect_inventory(checkpoint_dir=tmp_path)) == []


def test_cli_emits_json_before_returning_failure(rubin_runtime, monkeypatch, capsys):
    monkeypatch.delenv("SGLANG_VERSION")
    assert runtime.main(["--validate", "--launcher-image", runtime.IMAGE_REF]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["validation"]["requested"] is True
    assert any("SGLANG_VERSION" in error for error in output["validation"]["errors"])


def test_cli_keeps_framework_diagnostics_out_of_json(rubin_runtime, monkeypatch, capsys):
    def import_module(name):
        print("framework initialization diagnostic")
        return rubin_runtime

    monkeypatch.setattr(runtime.importlib, "import_module", import_module)
    assert runtime.main(["--validate"]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["validation"]["errors"] == []
    assert "framework initialization diagnostic" in output.err


def test_cli_redirects_native_import_output_and_restores_stdout(tmp_path):
    (tmp_path / "torch.py").write_text(
        "import os\n"
        "from types import SimpleNamespace\n"
        "os.write(1, b'native import diagnostic\\n')\n"
        "print('Python import diagnostic')\n"
        "__version__ = 'test'\n"
        "version = SimpleNamespace(cuda=None)\n"
        "cuda = SimpleNamespace(is_available=lambda: False)\n"
    )
    result = _runtime_subprocess(
        tmp_path,
        """
        result = runtime.main([])
        sys.stdout.flush()
        os.write(1, b'AFTER_MAIN\\n')
        raise SystemExit(result)
        """,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.endswith("AFTER_MAIN\n")
    inventory = json.loads(result.stdout.removesuffix("AFTER_MAIN\n"))
    assert inventory["observed"]["cuda"]["torch_version"] == "test"
    assert "native import diagnostic" in result.stderr
    assert "Python import diagnostic" in result.stderr


def test_cli_restores_stdout_when_inventory_raises(tmp_path):
    result = _runtime_subprocess(
        tmp_path,
        """
        import ctypes
        libc = ctypes.CDLL(None)

        def fail(**kwargs):
            os.write(1, b'native diagnostic before failure\\n')
            libc.printf(b'buffered native diagnostic before failure\\n')
            raise RuntimeError('probe failed')

        runtime.collect_inventory = fail
        try:
            runtime.main([])
        except RuntimeError:
            libc.printf(b'RESTORED\\n')
        else:
            raise AssertionError('inventory failure was swallowed')
        """,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "RESTORED\n"
    assert result.stderr == "native diagnostic before failure\nbuffered native diagnostic before failure\n"


def test_cli_flushes_buffered_native_output_before_redirect_and_restore(tmp_path):
    result = _runtime_subprocess(
        tmp_path,
        """
        import ctypes
        libc = ctypes.CDLL(None)
        libc.printf(b'BEFORE_MAIN\\n')

        def import_module(name):
            libc.printf(b'buffered native import diagnostic\\n')
            raise ImportError('native import failed')

        runtime.importlib.import_module = import_module
        result = runtime.main(['--check-imports', '--validate'])
        sys.stdout.flush()
        libc.printf(b'AFTER_MAIN\\n')
        raise SystemExit(result)
        """,
    )

    assert result.returncode == 1, result.stderr
    assert result.stdout.startswith("BEFORE_MAIN\n")
    assert result.stdout.endswith("AFTER_MAIN\n")
    inventory = json.loads(result.stdout.removeprefix("BEFORE_MAIN\n").removesuffix("AFTER_MAIN\n"))
    assert inventory["observed"]["cuda"]["error"] == "ImportError: native import failed"
    assert inventory["observed"]["imports"]["sglang"]["error"] == "ImportError: native import failed"
    assert result.stderr == "buffered native import diagnostic\n" * 4
