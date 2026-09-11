FROM nvcr.io/nvidian/dynamo-dev/vllm-agentx@sha256:b58174af2daaf4d02c4845fc90dd3f18615fc7c3ef930fb852ed0d29e76f1d12
USER root
COPY --chmod=0644 update-fpm.patch parent.sha256 /opt/fpm/
COPY --chmod=0644 test_forward_pass_metrics.py test_shm_broadcast.py /opt/fpm/tests/
COPY --chmod=0644 forward_pass_metrics_subscriber.py /opt/fpm/
RUN uv pip install --python /opt/fpm/.venv/bin/python 'multiprocess==0.70.18'
RUN set -eu; \
    VROOT=$(/opt/fpm/.venv/bin/python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("vllm").submodule_search_locations)))'); \
    cd "$(dirname "$VROOT")"; \
    sha256sum -c /opt/fpm/parent.sha256; \
    patch --batch --fuzz=0 -p1 < /opt/fpm/update-fpm.patch; \
    /opt/fpm/.venv/bin/python -m compileall -q "$VROOT/v1/metrics/forward_pass_metrics.py"
LABEL ai.dynamo.fpm.pr.revision="b3563fc65ae0f5359802593d78e7ea097e1fed31" \
      ai.dynamo.fpm.timing_scope="model_step_cuda" \
      ai.dynamo.fpm.vllm_base="2cf0a6915ce544dc493a0990f2ea38d81601128a" \
      ai.dynamo.fpm.parent="b58174af2daaf4d02c4845fc90dd3f18615fc7c3ef930fb852ed0d29e76f1d12"
USER dynamo
CMD ["bash"]
