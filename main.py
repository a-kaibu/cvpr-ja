from __future__ import annotations

import argparse
import difflib
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_MARKDOWN_PATH = ROOT / "2025.md"
CACHE_ROOT = ROOT / "cache" / "2025"
ARXIV_CACHE_DIR = CACHE_ROOT / "arxiv_records"
TRANSLATION_CACHE_DIR = CACHE_ROOT / "translations"
USER_AGENT = "cvpr-ja/0.1 (+https://arxiv.org)"
LIST_PREFIXES = ("- arXiv:", "- PDF:", "- Summary:", "- Summary (JA):")
ARXIV_NAMESPACES = {"atom": "http://www.w3.org/2005/Atom"}
NO_MATCH = "__NO_MATCH__"


@dataclass(frozen=True)
class PaperEntry:
    paper_id: str
    title: str
    start: int
    end: int


@dataclass(frozen=True)
class ArxivRecord:
    abs_url: str
    pdf_url: str
    summary: str
    matched_title: str
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append arXiv abstract links, PDF links, summaries, and Japanese "
            "translations to 2025.md."
        )
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=DEFAULT_MARKDOWN_PATH,
        help="Path to the markdown file to enrich.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="1-based paper index to start from.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of papers to process.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.9,
        help="Minimum normalized title similarity required to accept a match.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=6,
        help="Maximum number of paper titles to include in a single arXiv query.",
    )
    parser.add_argument(
        "--batch-bytes",
        type=int,
        default=1800,
        help="Approximate maximum encoded query size per arXiv batch.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=3.0,
        help="Delay between batched arXiv requests.",
    )
    parser.add_argument(
        "--translation-workers",
        type=int,
        default=4,
        help="Number of concurrent workers for summary translation.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Re-fetch even when the markdown already contains arXiv fields.",
    )
    return parser.parse_args()


def normalize_title(text: str) -> str:
    lowered = html.unescape(text).lower()
    lowered = lowered.replace("’", "'").replace("`", "'")
    return re.sub(r"[^a-z0-9]+", "", lowered)


def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def request_text(url: str, *, timeout: int = 30) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def read_cached_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_entries(markdown: str) -> list[PaperEntry]:
    pattern = re.compile(r"^## (?:⭐ )?(?P<id>\d{4})\. (?P<title>.+)$", re.MULTILINE)
    matches = list(pattern.finditer(markdown))
    entries: list[PaperEntry] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        entries.append(
            PaperEntry(
                paper_id=match.group("id"),
                title=match.group("title").strip(),
                start=start,
                end=end,
            )
        )
    return entries


def needs_update(block: str) -> bool:
    return any(prefix not in block for prefix in LIST_PREFIXES)


def arxiv_cache_path(entry: PaperEntry) -> Path:
    return ARXIV_CACHE_DIR / f"{entry.paper_id}.json"


def translation_cache_path(entry: PaperEntry) -> Path:
    return TRANSLATION_CACHE_DIR / f"{entry.paper_id}.json"


def load_cached_record(entry: PaperEntry) -> ArxivRecord | None | str:
    cache_path = arxiv_cache_path(entry)
    cached = read_cached_text(cache_path)
    if cached is None:
        return None

    try:
        payload = json.loads(cached)
    except json.JSONDecodeError:
        return None

    if payload.get("status") == NO_MATCH:
        return NO_MATCH

    return ArxivRecord(
        abs_url=payload["abs_url"],
        pdf_url=payload["pdf_url"],
        summary=payload["summary"],
        matched_title=payload["matched_title"],
        score=float(payload["score"]),
    )


def save_cached_record(entry: PaperEntry, record: ArxivRecord | None) -> None:
    cache_path = arxiv_cache_path(entry)
    if record is None:
        payload = {"status": NO_MATCH, "title": entry.title}
    else:
        payload = {"status": "ok", **asdict(record), "title": entry.title}
    write_text(cache_path, json.dumps(payload, ensure_ascii=False, indent=2))


def parse_feed_records(xml_text: str) -> list[ArxivRecord]:
    root = ET.fromstring(xml_text)
    records: list[ArxivRecord] = []

    for candidate in root.findall("atom:entry", ARXIV_NAMESPACES):
        title = clean_whitespace(
            candidate.findtext("atom:title", default="", namespaces=ARXIV_NAMESPACES)
        )
        summary = clean_whitespace(
            candidate.findtext("atom:summary", default="", namespaces=ARXIV_NAMESPACES)
        )
        if not title or not summary:
            continue

        abs_url = clean_whitespace(
            candidate.findtext("atom:id", default="", namespaces=ARXIV_NAMESPACES)
        ).replace("http://", "https://")
        pdf_url = ""
        for link in candidate.findall("atom:link", ARXIV_NAMESPACES):
            if link.attrib.get("title") == "pdf":
                pdf_url = clean_whitespace(link.attrib.get("href", ""))
                break
        if not pdf_url and abs_url:
            pdf_url = abs_url.replace("/abs/", "/pdf/")

        records.append(
            ArxivRecord(
                abs_url=abs_url,
                pdf_url=pdf_url,
                summary=summary,
                matched_title=title,
                score=0.0,
            )
        )

    return records


def match_records_to_entries(
    batch: list[PaperEntry],
    records: list[ArxivRecord],
    min_score: float,
) -> dict[str, ArxivRecord | None]:
    matched: dict[str, ArxivRecord | None] = {}
    used_indices: set[int] = set()

    for entry in batch:
        expected = normalize_title(entry.title)
        best_index = -1
        best_record: ArxivRecord | None = None
        best_score = 0.0

        for index, record in enumerate(records):
            if index in used_indices:
                continue
            score = difflib.SequenceMatcher(
                None, expected, normalize_title(record.matched_title)
            ).ratio()
            if score > best_score:
                best_index = index
                best_record = record
                best_score = score

        if best_record is not None and best_score >= min_score:
            used_indices.add(best_index)
            matched[entry.paper_id] = ArxivRecord(
                abs_url=best_record.abs_url,
                pdf_url=best_record.pdf_url,
                summary=best_record.summary,
                matched_title=best_record.matched_title,
                score=best_score,
            )
        else:
            matched[entry.paper_id] = None

    return matched


def encoded_title_clause(title: str) -> str:
    return f'ti:"{title.replace(chr(34), "")}"'


def build_batches(
    entries: list[PaperEntry],
    *,
    batch_size: int,
    batch_bytes: int,
) -> list[list[PaperEntry]]:
    batches: list[list[PaperEntry]] = []
    current: list[PaperEntry] = []
    current_size = 0

    for entry in entries:
        clause = encoded_title_clause(entry.title)
        clause_size = len(urllib.parse.quote_plus(clause))
        separator_size = len(urllib.parse.quote_plus(" OR ")) if current else 0
        would_exceed = current and (
            len(current) >= batch_size or current_size + separator_size + clause_size > batch_bytes
        )

        if would_exceed:
            batches.append(current)
            current = []
            current_size = 0
            separator_size = 0

        current.append(entry)
        current_size += separator_size + clause_size

    if current:
        batches.append(current)

    return batches


def query_arxiv_batch(
    batch: list[PaperEntry], *, min_score: float
) -> dict[str, ArxivRecord | None] | None:
    query = " OR ".join(encoded_title_clause(entry.title) for entry in batch)
    url = (
        "https://export.arxiv.org/api/query?"
        + urllib.parse.urlencode(
            {
                "search_query": query,
                "start": 0,
                "max_results": max(len(batch) * 8, 10),
            }
        )
    )

    last_error: Exception | None = None
    for attempt in range(4):
        try:
            xml_text = request_text(url)
            records = parse_feed_records(xml_text)
            return match_records_to_entries(batch, records, min_score)
        except ET.ParseError as exc:
            last_error = exc
            break
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429:
                time.sleep((attempt + 1) * 5)
                continue
            break
        except urllib.error.URLError as exc:
            last_error = exc
            time.sleep((attempt + 1) * 3)

    paper_ids = ", ".join(entry.paper_id for entry in batch)
    print(f"[warn] arXiv batch failed for {paper_ids}: {last_error}", file=sys.stderr)
    return None


def fetch_missing_records(
    entries: list[PaperEntry],
    *,
    min_score: float,
    batch_size: int,
    batch_bytes: int,
    pause_seconds: float,
) -> dict[str, ArxivRecord | None]:
    fetched: dict[str, ArxivRecord | None] = {}
    batches = build_batches(entries, batch_size=batch_size, batch_bytes=batch_bytes)

    for index, batch in enumerate(batches, start=1):
        matches = query_arxiv_batch(batch, min_score=min_score)
        if matches is None:
            print(
                f"[info] arXiv batch {index}/{len(batches)} deferred due to request failure",
                file=sys.stderr,
            )
            if index != len(batches):
                time.sleep(max(pause_seconds, 0))
            continue

        for entry in batch:
            record = matches.get(entry.paper_id)
            save_cached_record(entry, record)
            fetched[entry.paper_id] = record

        print(
            f"[info] arXiv batch {index}/{len(batches)} processed "
            f"({len(batch)} papers, {sum(value is not None for value in matches.values())} matches)",
            file=sys.stderr,
        )
        if index != len(batches):
            time.sleep(max(pause_seconds, 0))

    return fetched


def fetch_translation(entry: PaperEntry, summary: str) -> str | None:
    cache_path = translation_cache_path(entry)
    cached = read_cached_text(cache_path)
    if cached is not None:
        try:
            payload = json.loads(cached)
            if payload.get("source") == summary:
                return payload.get("translated")
        except json.JSONDecodeError:
            pass

    url = (
        "https://translate.googleapis.com/translate_a/single?"
        + urllib.parse.urlencode(
            {
                "client": "gtx",
                "sl": "en",
                "tl": "ja",
                "dt": "t",
                "q": summary,
            }
        )
    )

    try:
        payload = json.loads(request_text(url))
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"[warn] translation failed for {entry.paper_id}: {exc}", file=sys.stderr)
        return None

    translated = "".join(chunk[0] for chunk in payload[0] if chunk and chunk[0]).strip()
    if not translated:
        return None

    write_text(
        cache_path,
        json.dumps(
            {"source": summary, "translated": translated},
            ensure_ascii=False,
            indent=2,
        ),
    )
    return translated


def enrich_block(block: str, record: ArxivRecord, summary_ja: str | None) -> str:
    lines = block.splitlines()
    kept_lines = [line for line in lines if not line.startswith(LIST_PREFIXES)]

    insert_at = len(kept_lines)
    for index, line in enumerate(kept_lines):
        if line.startswith("- "):
            insert_at = index + 1
        elif insert_at != len(kept_lines):
            break

    extra_lines = [
        f"- arXiv: {record.abs_url}",
        f"- PDF: {record.pdf_url}",
        f"- Summary: {record.summary}",
    ]
    if summary_ja:
        extra_lines.append(f"- Summary (JA): {summary_ja}")

    updated_lines = kept_lines[:insert_at] + extra_lines + kept_lines[insert_at:]
    return "\n".join(updated_lines).rstrip() + "\n\n"


def enrich_markdown(
    original_markdown: str,
    entries: list[PaperEntry],
    blocks: dict[str, str],
    records_by_id: dict[str, ArxivRecord | None],
    summary_ja_by_id: dict[str, str | None],
) -> tuple[str, int]:
    updated_count = 0
    for entry in entries:
        record = records_by_id.get(entry.paper_id)
        if record is None:
            continue
        new_block = enrich_block(blocks[entry.paper_id], record, summary_ja_by_id.get(entry.paper_id))
        if new_block != blocks[entry.paper_id]:
            updated_count += 1
            blocks[entry.paper_id] = new_block

    rebuilt = []
    cursor = 0
    for entry in entries:
        rebuilt.append(original_markdown[cursor : entry.start])
        rebuilt.append(blocks[entry.paper_id])
        cursor = entry.end
    rebuilt.append(original_markdown[cursor:])
    return "".join(rebuilt), updated_count


def main() -> int:
    args = parse_args()
    markdown_path = args.markdown.resolve()
    original_markdown = markdown_path.read_text(encoding="utf-8")
    entries = parse_entries(original_markdown)

    start_index = max(args.start - 1, 0)
    selected_entries = entries[start_index:]
    if args.limit is not None:
        selected_entries = selected_entries[: args.limit]

    blocks = {entry.paper_id: original_markdown[entry.start : entry.end] for entry in entries}
    target_entries = [
        entry
        for entry in selected_entries
        if args.overwrite_existing or needs_update(blocks[entry.paper_id])
    ]

    records_by_id: dict[str, ArxivRecord | None] = {}
    missing_entries: list[PaperEntry] = []
    for entry in target_entries:
        cached = load_cached_record(entry)
        if cached == NO_MATCH:
            records_by_id[entry.paper_id] = None
        elif isinstance(cached, ArxivRecord):
            records_by_id[entry.paper_id] = cached
        else:
            missing_entries.append(entry)

    if missing_entries:
        fetched = fetch_missing_records(
            missing_entries,
            min_score=args.min_score,
            batch_size=max(args.batch_size, 1),
            batch_bytes=max(args.batch_bytes, 400),
            pause_seconds=args.pause_seconds,
        )
        records_by_id.update(fetched)

    summary_ja_by_id: dict[str, str | None] = {}
    translation_targets = [
        entry
        for entry in target_entries
        if records_by_id.get(entry.paper_id) is not None
    ]
    with ThreadPoolExecutor(max_workers=max(args.translation_workers, 1)) as executor:
        futures = {
            executor.submit(
                fetch_translation,
                entry,
                records_by_id[entry.paper_id].summary,  # type: ignore[union-attr]
            ): entry
            for entry in translation_targets
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            entry = futures[future]
            try:
                summary_ja_by_id[entry.paper_id] = future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] translation task failed for {entry.paper_id}: {exc}", file=sys.stderr)
                summary_ja_by_id[entry.paper_id] = None

            if completed % 50 == 0 or completed == len(futures):
                print(
                    f"[info] translated {completed}/{len(futures)} summaries",
                    file=sys.stderr,
                )

    updated_markdown, updated_count = enrich_markdown(
        original_markdown,
        entries,
        blocks,
        records_by_id,
        summary_ja_by_id,
    )

    if updated_markdown != original_markdown:
        markdown_path.write_text(updated_markdown, encoding="utf-8")

    print(f"Updated {updated_count} paper entries in {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
