FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y sudo wget ca-certificates git xz-utils

COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /uvx /bin/
RUN uv python install -i /tmp/python-download 3.12.11 \
    && cp -r /tmp/python-download/cpython-3.12.11-linux-x86_64-gnu/* /usr/local/ \
    && rm -rf /tmp/python-download \
    && rm /usr/local/lib/python3.12/EXTERNALLY-MANAGED

COPY requirements.txt /requirements.txt
RUN uv pip install --system --torch-backend=auto -r /requirements.txt

WORKDIR /workspace

ENTRYPOINT []
CMD ["/bin/bash"]
