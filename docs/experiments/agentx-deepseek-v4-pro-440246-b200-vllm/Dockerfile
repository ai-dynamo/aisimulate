FROM nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly@sha256:31e6a99a9a25be541ffdffabb31eb72747759f82647924067fe08204e5c2c0af
USER root
COPY --chmod=0644 native-fpm.patch base.sha256 /opt/fpm/
COPY --chmod=0644 test_forward_pass_metrics.py /opt/fpm/tests/test_forward_pass_metrics.py
COPY --chmod=0644 forward_pass_metrics_subscriber.py record_fpm.py /opt/fpm/
RUN uv venv --system-site-packages /opt/fpm/.venv \
    && uv pip install --python /opt/fpm/.venv/bin/python pytest
RUN set -eu; \
    VROOT=$(/opt/fpm/.venv/bin/python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("vllm").submodule_search_locations)))'); \
    cd "$(dirname "$VROOT")"; \
    sha256sum -c /opt/fpm/base.sha256; \
    patch --batch --fuzz=0 -p1 < /opt/fpm/native-fpm.patch; \
    /opt/fpm/.venv/bin/python -m compileall -q "$VROOT/v1/metrics/forward_pass_metrics.py"
RUN uv venv /opt/agentx-aiperf \
    && uv pip install --python /opt/agentx-aiperf/bin/python \
       'aiperf @ https://github.com/SemiAnalysisAI/aiperf/archive/754356e9a39acc6cc6afb242d123bb57c3fb6f75.tar.gz' \
       'vllm-router==0.1.14' \
    && uv pip check --python /opt/agentx-aiperf/bin/python \
    && uv pip freeze --python /opt/agentx-aiperf/bin/python > /opt/agentx-aiperf/freeze.txt
ARG FPM_PR_REVISION=unpublished-sd-fix
LABEL ai.dynamo.fpm.method="native-vllm-pr-52061" \
      ai.dynamo.fpm.pr.revision="${FPM_PR_REVISION}" \
      ai.dynamo.fpm.timing_scope="model_step_cuda" \
      ai.dynamo.fpm.vllm_base="2cf0a6915ce544dc493a0990f2ea38d81601128a"
USER dynamo
CMD ["bash"]
