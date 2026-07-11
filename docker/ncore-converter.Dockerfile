ARG BASE_IMAGE=nsb/ncore-conda-base:2026-07-10
FROM ${BASE_IMAGE}

WORKDIR /workspace

COPY third_party/ncore_converter /workspace/third_party/ncore_converter
COPY scripts/validate_ncore_env.py /workspace/scripts/validate_ncore_env.py

ENV PYTHONPATH=/workspace/third_party/ncore_converter

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "scripts/validate_ncore_env.py", "--converter-root", "/workspace/third_party/ncore_converter"]
