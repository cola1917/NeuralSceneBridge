FROM ubuntu:22.04

ARG MINICONDA_VERSION=py311_25.5.1-0
ARG CONDA_DIR=/opt/conda
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_DEFAULT_TIMEOUT=120
ARG PIP_RETRIES=10

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT}
ENV PIP_RETRIES=${PIP_RETRIES}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        tini \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL \
        "https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" \
        -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniconda.sh \
    && conda config --system --set auto_update_conda false \
    && conda clean -afy

COPY environment/ncore-converter.yml /tmp/ncore-converter.yml
COPY requirements/ncore-converter.txt /tmp/ncore-converter.txt

RUN conda env create -f /tmp/ncore-converter.yml \
    && conda run -n ncore-converter python -m pip install --no-cache-dir -r /tmp/ncore-converter.txt \
    && conda clean -afy

ENV CONDA_DEFAULT_ENV=ncore-converter
ENV PATH=${CONDA_DIR}/envs/ncore-converter/bin:${CONDA_DIR}/bin:${PATH}

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "--version"]
