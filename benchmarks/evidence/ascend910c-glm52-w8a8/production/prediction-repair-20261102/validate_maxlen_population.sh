#!/usr/bin/env bash
set -euo pipefail
export CARGO_HOME=/workspace/build/node20-toolchain/cargo
export RUSTUP_HOME=/workspace/build/node20-toolchain/rustup
export PATH="$CARGO_HOME/bin:$PATH"
export CARGO_BUILD_JOBS=3
validation_dir=/workspace/benchmarks/evidence/ascend910c-glm52-w8a8/production/prediction-repair-20261102/maxlen-population-validation-20261104
cd /workspace
rustfmt --edition 2024 crates/core/src/perfmodel/engine/ascend_dp.rs
cargo test --offline -p aisimulate-core --lib perfmodel::engine::ascend_dp::tests > "$validation_dir/native-tests.log" 2>&1
cd /workspace/python/aisimulate
maturin build --release --offline -o /workspace/build/maxlen-population-20261104 > "$validation_dir/wheel-build.log" 2>&1
python3 -m pip install --no-deps --force-reinstall /workspace/build/maxlen-population-20261104/*.whl > "$validation_dir/wheel-install.log" 2>&1
cd /workspace
python3 -m pytest -c python/aisimulate/pytest.ini -o addopts="" -o timeout=600 -p no:cacheprovider crates/core/parity_tests/perfmodel/test_engine_step_parity.py crates/core/parity_tests/perfmodel/test_compile_engine_parity.py python/aisimulate/tests/unit/sdk/models/test_ascend_glm52.py python/aisimulate/tests/cross_package/test_single_oracle_contract.py python/aisimulate/tests/unit/collector/test_ascend_layer_population_export.py -q > "$validation_dir/parity-tests.log" 2>&1
