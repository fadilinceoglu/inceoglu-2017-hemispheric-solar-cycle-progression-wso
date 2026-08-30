#!/usr/bin/env python3
"""Download and verify the WSO filled synoptic maps used by the study.

The observations are written only to the ignored ``.cache/wso/`` tree.  A
manifest records the authoritative URL, byte count, and SHA-256 checksum of
every response.  No downloaded map is copied into the repository data tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inceoglu2017.io import sha256_bytes  # noqa: E402
from inceoglu2017.paths import (  # noqa: E402
    FIRST_CARRINGTON_ROTATION,
    LAST_CARRINGTON_ROTATION,
    WSO_MANIFEST,
    WSO_MAPS,
    WSO_URL_TEMPLATE,
)
from inceoglu2017.wso import parse_wso_filled_map  # noqa: E402


# These anchors and the complete manifest digest define the expected WSO input
# identities for the paper interval. A mismatch is surfaced rather than used.
EXPECTED_FILE_SHA256 = {
    1642: "c0061f20e91e529f9e68ac0758fbb5be17d9b3e7297145f36cd15232e237fc98",
    2149: "ce0a1f102bee6ac00e27b1e2222a27246db5cee26a0d5417914421010655c2f1",
    2150: "2ddc7a52e653ae05d249414891f3990a18c22efaf96f0e1b7c9c55bc837f2b79",
    2155: "11be9f6c5b07555ee4dbae4f6c32ff343057acefc538d09158e3699457d49f99",
    2164: "7ec9c55bb3a954afdbc4403da0e3ea6128f6263c169f871fc1a65097141d2fb9",
    2182: "76f599dbcfe6157d1c8a94c6c67da8f347f788902b662fa95ee45c08000f06d7",
}
EXPECTED_CR1642_2182_MANIFEST_SHA256 = (
    "e6188ead532be235cdb70952591b5363273e1b04bd674f0bfdd7d847aea4ed46"
)


def validate_map(data: bytes, rotation: int) -> None:
    """Require an ASCII, filled, exact 72x30 map for *rotation*."""

    if b"filled_synop" not in data[:256]:
        raise ValueError(f"CR {rotation}: response is not a WSO filled synoptic map")
    if b"XXX" in data:
        raise ValueError(f"CR {rotation}: unfilled missing values found")
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"CR {rotation}: response is not ASCII") from error
    parse_wso_filled_map(text, target_rotation=rotation)


def manifest_digest(records: list[dict[str, object]]) -> str:
    """Hash a conventional checksum manifest independently of JSON metadata."""

    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: int(item["cr"])):
        digest.update(f"{record['sha256']}  {record['file']}\n".encode("ascii"))
    return digest.hexdigest()


def _download(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "inceoglu-2017-reproduction/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise OSError(f"HTTP {response.status} for {url}")
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-cr", type=int, default=FIRST_CARRINGTON_ROTATION)
    parser.add_argument("--last-cr", type=int, default=LAST_CARRINGTON_ROTATION)
    parser.add_argument("--destination", type=Path, default=WSO_MAPS)
    parser.add_argument("--manifest", type=Path, default=WSO_MANIFEST)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not (
        FIRST_CARRINGTON_ROTATION
        <= args.first_cr
        <= args.last_cr
        <= LAST_CARRINGTON_ROTATION
    ):
        raise ValueError(
            f"Require {FIRST_CARRINGTON_ROTATION} <= first CR <= last CR "
            f"<= {LAST_CARRINGTON_ROTATION}"
        )
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")

    args.destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for rotation in range(args.first_cr, args.last_cr + 1):
        name = f"WSO.{rotation}.F.txt"
        destination = args.destination / name
        url = WSO_URL_TEMPLATE.format(cr=rotation)
        if destination.exists() and not args.force:
            data = destination.read_bytes()
        else:
            data = _download(url, timeout=args.timeout)
            validate_map(data, rotation)
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(data)
            temporary.replace(destination)

        validate_map(data, rotation)
        checksum = sha256_bytes(data)
        expected = EXPECTED_FILE_SHA256.get(rotation)
        if expected is not None and checksum != expected:
            raise ValueError(
                f"CR {rotation}: SHA-256 {checksum} differs from expected {expected}"
            )
        records.append(
            {
                "cr": rotation,
                "file": name,
                "url": url,
                "bytes": len(data),
                "sha256": checksum,
            }
        )

    checksum_manifest = manifest_digest(records)
    complete_interval = (
        args.first_cr == FIRST_CARRINGTON_ROTATION
        and args.last_cr == LAST_CARRINGTON_ROTATION
    )
    if complete_interval and checksum_manifest != EXPECTED_CR1642_2182_MANIFEST_SHA256:
        raise ValueError(
            "The complete CR 1642--2182 checksum manifest differs from the "
            "expected release manifest"
        )

    manifest = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Wilcox Solar Observatory filled synoptic maps",
        "url_template": WSO_URL_TEMPLATE,
        "first_carrington_rotation": args.first_cr,
        "last_carrington_rotation": args.last_cr,
        "checksum_manifest_sha256": checksum_manifest,
        "rights": (
            "WSO requests acknowledgement, notification, and copies of resulting "
            "reports; no open redistribution license is stated."
        ),
        "records": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = args.manifest.with_suffix(args.manifest.suffix + ".part")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(args.manifest)
    print(
        f"Verified {len(records)} WSO maps in {args.destination}; "
        f"manifest SHA-256 {checksum_manifest}"
    )


if __name__ == "__main__":
    main()
