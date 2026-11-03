import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

baseline = Path('/private/tmp/gbdt-baseline-gym-a8b30ca6')
sys.path.insert(0, str(baseline))
label, output, *test_args = sys.argv[1:]
output = Path(output)
output.mkdir(parents=True, exist_ok=True)
modules = {}
for name in ('scripts.fpm_accuracy.models.learned_scenario', 'scripts.fpm_accuracy.scenario_contract', 'aisimulate_core.sdk.rust_engine_step', 'aisimulate_core.sdk.fpm_learned', 'aisimulate._runtime', 'sklearn'):
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    modules[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
assert modules['scripts.fpm_accuracy.models.learned_scenario']['path'].startswith(str(baseline) + '/')
expected = (os.environ['FPM_COMPAT_EXPECTED_SDK_ROOT'], os.environ['FPM_COMPAT_EXPECTED_NATIVE_SHA'])
for name in ('aisimulate_core.sdk.rust_engine_step', 'aisimulate_core.sdk.fpm_learned', 'aisimulate._runtime'):
    assert modules[name]['path'].startswith(expected[0]), (name, modules[name])
assert modules['aisimulate._runtime']['sha256'] == expected[1]
record = {'label': label, 'python': sys.executable, 'argv': sys.argv, 'cwd': os.getcwd(), 'modules': modules, 'threads': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}}
(output / f'{label}-origins.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
import pytest
raise SystemExit(pytest.main(test_args))
