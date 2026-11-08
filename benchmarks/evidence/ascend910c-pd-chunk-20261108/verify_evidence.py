"""Verify the frozen archive and every member without extracting files."""
import hashlib
import json
from pathlib import Path
import tarfile


root = Path(__file__).resolve().parent
manifest = json.loads((root/'manifest.json').read_text())
archive = root/manifest['archive']
with archive.open('rb') as content:
    archive_sha256 = hashlib.file_digest(content, 'sha256').hexdigest()
assert archive_sha256 == manifest['archive_sha256']
expected = {row['member']: row for row in manifest['members']}
assert len(expected) == len(manifest['members'])
verified = set()
with tarfile.open(archive, 'r|xz') as stream:
    for member in stream:
        assert member.isfile() and member.name not in verified
        entry = expected[member.name]
        assert member.size == entry['bytes']
        with stream.extractfile(member) as content:
            digest = hashlib.file_digest(content, 'sha256').hexdigest()
        assert digest == entry['sha256'], member.name
        verified.add(member.name)
assert verified == set(expected)
result = dict(status='verified', archive=archive.name,
              archive_sha256=archive_sha256, verified_members=len(verified))
(root/'results/archive-verification.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
