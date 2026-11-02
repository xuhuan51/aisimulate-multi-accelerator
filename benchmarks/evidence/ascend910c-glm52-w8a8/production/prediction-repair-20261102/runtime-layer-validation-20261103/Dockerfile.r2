FROM aisimulate-910c:causal-repair-20261102
COPY aisimulate-0.13.0-cp311-abi3-linux_aarch64.whl /tmp/aisimulate.whl
RUN mv /tmp/aisimulate.whl /tmp/aisimulate-0.13.0-cp311-abi3-linux_aarch64.whl && python3 -m pip install --no-deps --force-reinstall /tmp/aisimulate-0.13.0-cp311-abi3-linux_aarch64.whl && rm /tmp/aisimulate-0.13.0-cp311-abi3-linux_aarch64.whl
