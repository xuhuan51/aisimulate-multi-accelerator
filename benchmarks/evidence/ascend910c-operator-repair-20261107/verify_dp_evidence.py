"""Requalify the frozen DP-buffer samples and check all57 published rows."""

import hashlib
import json
import tarfile
import tempfile
from pathlib import Path

import pyarrow as arrow
import pyarrow.parquet as parquet
from collector.sglang.ascend_vector_export import validated_rows


def main():
    here = Path(__file__).resolve().parent
    repository = here.parents[2]
    directory = repository / "python/aisimulate/src/aisimulate_core/systems/data/ascend910c/vector/sglang/0.5.22"
    metadata = json.loads((directory / "metadata/dp-buffer-20261107.json").read_text())
    table = parquet.read_table(directory / "ascend_vector_perf.parquet")
    expected = []
    with tempfile.TemporaryDirectory() as temporary:
        evidence = Path(temporary)
        with tarfile.open(here / "dp-qualified-evidence.tar.gz") as archive:
            for member in archive.getmembers():
                if not member.isfile() or Path(member.name).name != member.name:
                    raise ValueError("Expected regular top-level evidence files")
                (evidence / member.name).write_bytes(archive.extractfile(member).read())
        for campaign in metadata["campaigns"]:
            observed = json.loads((evidence / campaign["observation"]).read_text())
            for name, filename in [("collector", "ascend_vector.py"), ("kernels", "ascend_vector_kernels.py")]:
                source = here / "measured-source" / filename
                assert hashlib.sha256(source.read_bytes()).hexdigest() == observed["collector_source_sha256"][name]
            rows, report = validated_rows(
                evidence / campaign["plan"],
                [evidence / campaign["observation"]],
                campaign["report"]["identity"]["runtime"],
            )
            assert report["qualified"] == campaign["report"]["qualified"]
            expected.extend(row for row in rows if row["role"] in campaign["published_roles"])
    roles = {"dp_buffer_zero", "dp_gather_copy", "dp_scatter_copy"}
    published = [row for row in table.to_pylist() if row["role"] in roles]
    converted = arrow.Table.from_pylist(expected, schema=table.schema).to_pylist()

    def key(row):
        return row["role"], row["tokens"]

    # Token-axis field names are checked through the current table contract.
    assert len(published) == len(converted) == 57
    assert sorted(published, key=key) == sorted(converted, key=key)
    print(
        json.dumps(
            {
                "status": "passed",
                "rows": 57,
                "scope": "Frozen measured source, raw sample qualification and published-row equality",
            }
        )
    )


if __name__ == "__main__":
    main()
