"""Download HTTP(S) links found in Excel workbooks under the data directory."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from openpyxl import load_workbook


URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def find_urls(workbook_path: Path) -> list[str]:
    """Return unique URLs from cell hyperlinks and cell text."""
    urls: list[str] = []
    seen: set[str] = set()
    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    try:
        for worksheet in workbook.worksheets:
            for row in worksheet.iter_rows():
                for cell in row:
                    candidates: list[str] = []
                    hyperlink = getattr(cell, "hyperlink", None)
                    if hyperlink and hyperlink.target:
                        candidates.append(hyperlink.target)
                    if isinstance(cell.value, str):
                        candidates.extend(URL_PATTERN.findall(cell.value))
                    for url in candidates:
                        url = url.rstrip(".,;:)]}")
                        if url.lower().startswith(("http://", "https://")) and url not in seen:
                            seen.add(url)
                            urls.append(url)
    finally:
        workbook.close()
    return urls


def safe_filename(url: str) -> str:
    """Choose a filesystem-safe name based on a URL."""
    parsed = urlparse(url)
    name = Path(unquote(parsed.path)).name or "download"
    name = INVALID_FILENAME_CHARS.sub("_", name).strip(" .") or "download"
    return name[:180]


def available_path(directory: Path, filename: str) -> Path:
    target = directory / filename
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    counter = 2
    while True:
        target = directory / f"{stem}_{counter}{suffix}"
        if not target.exists():
            return target
        counter += 1


def download(url: str, output_directory: Path) -> Path:
    request = Request(url, headers={"User-Agent": "excel-link-downloader/1.0"})
    with urlopen(request, timeout=60) as response:
        filename = safe_filename(url)
        content_disposition = response.headers.get_filename()
        if content_disposition:
            filename = INVALID_FILENAME_CHARS.sub("_", content_disposition).strip(" .")
        destination = available_path(output_directory, filename)
        with destination.open("wb") as output_file:
            while chunk := response.read(1024 * 1024):
                output_file.write(chunk)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing .xlsx files")
    parser.add_argument("--output-dir", type=Path, default=Path("downloaded"), help="Directory for downloaded files")
    parser.add_argument("--dry-run", action="store_true", help="List links without downloading them")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workbook_paths = sorted(args.data_dir.glob("*.xlsx"))
    if not workbook_paths:
        print(f"No .xlsx files found in {args.data_dir}", file=sys.stderr)
        return 1

    all_urls: list[str] = []
    seen: set[str] = set()
    for workbook_path in workbook_paths:
        try:
            urls = find_urls(workbook_path)
        except Exception as error:
            print(f"ERROR {workbook_path.name}: {error}", file=sys.stderr)
            continue
        print(f"{workbook_path.name}: found {len(urls)} link(s)")
        for url in urls:
            if url not in seen:
                seen.add(url)
                all_urls.append(url)

    print(f"Total unique links: {len(all_urls)}")
    if args.dry_run:
        print("\n".join(all_urls))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for url in all_urls:
        try:
            destination = download(url, args.output_dir)
            print(f"Downloaded: {url} -> {destination.name}")
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            failures += 1
            print(f"FAILED: {url} ({error})", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())