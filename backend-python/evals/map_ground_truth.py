"""Map human source citations to verified PostgreSQL chunk IDs.

This utility never calls vector search, keyword search, an application LLM, or
an evaluator LLM. Each answerable question has explicit evidence phrases that
must occur on its human-cited source page. The phrases make the mapping easy to
review and keep retrieval output from becoming circular ground truth.
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.documents import get_document_chunks


DEFAULT_DATASET = Path("evals/datasets/v1/golden.jsonl")
DEFAULT_OUTPUT = Path("evals/datasets/v1/golden.mapped.jsonl")
DEFAULT_REPORT = Path("evals/results/ground_truth_mapping_report.json")
NORTHSTAR_SOURCE = "Northstar_RAG_Torture_Corpus_v2.pdf"


@dataclass(frozen=True)
class EvidenceRequirement:
    """Describe one piece of evidence that must exist in one stored chunk."""

    page: int
    phrase: str


def requirement(page: int, phrase: str) -> EvidenceRequirement:
    """Keep the large evidence table below compact and readable."""

    return EvidenceRequirement(page=page, phrase=phrase)


# These phrases were selected by inspecting the 58 stored Northstar chunks.
# Every phrase is scoped to its human-cited page. Multiple entries mean that
# every fact is required; they may resolve to the same chunk or different ones.
EVIDENCE_RULES: dict[str, list[EvidenceRequirement]] = {
    "Q01": [requirement(2, "Nova Edge is not an upgrade of Nova Plus; it is a deployment profile for intermittently connected sites")],
    "Q02": [
        requirement(2, "The internal product code NOVA-E refers to Nova Enterprise"),
        requirement(2, "The code NOVA-EDGE refers to Nova Edge"),
    ],
    "Q03": [
        requirement(3, "Nova Plus includes 75 human seats"),
        requirement(3, "uses the maximum number of concurrently licensed human seats recorded at the tenant daily snapshot"),
    ],
    "Q04": [requirement(3, "Service accounts, machine identities, and API keys do not consume human seats")],
    "Q05": [
        requirement(3, "Nova Plus allows at most 12 active API keys"),
        requirement(3, "Existing inactive or revoked keys do not count against the 12-key limit"),
    ],
    "Q06": [requirement(4, "This exclusion applies only to the Compliance Export API, not to the general REST API")],
    "Q07": [
        requirement(4, "The Compliance Export API is internally tagged API-CX9"),
        requirement(4, "API-C9X belongs to a retired prototype"),
    ],
    "Q08": [requirement(5, "M-482 complete + Enterprise EU + synchronous Compliance Export 45 s")],
    "Q09": [
        requirement(5, "The M-482 exception applies only after the tenant migration status is COMPLETE"),
        requirement(5, "SCHEDULED or IN_PROGRESS remains on its pre-migration timeout"),
    ],
    "Q10": [
        requirement(6, "Before 15 February 2026, Nova Enterprise EU tenants used a 30-second synchronous Compliance Export timeout during the M-482 pilot"),
        requirement(6, "That pilot value was superseded when the migration entered general availability"),
        requirement(5, "M-482 complete + Enterprise EU + synchronous Compliance Export 45 s"),
    ],
    "Q11": [requirement(7, "On Gateway 8.4 with TLS passthrough enabled, ERR-4812 means the presented certificate chain exceeds the configured verification depth")],
    "Q12": [requirement(7, "On Gateway 8.3 and earlier, ERR-4812 meant upstream handshake timeout")],
    "Q13": [requirement(7, "Increasing the network request timeout does not address this condition")],
    "Q15": [requirement(8, "LNM and private endpoints cannot be active at the same time")],
    "Q17": [requirement(9, "Status EXEMPT means the tenant remains governed by its contract-specific retention value; it does not automatically mean 60 days")],
    "Q18": [
        requirement(10, "Enterprise Compliance Export jobs may run for 90 minutes"),
        requirement(10, "synchronous timeout from section 4 must not be applied to an asynchronous Compliance Export job"),
    ],
    "Q19": [requirement(10, "Job code JOB-902 means execution window exceeded")],
    "Q20": [requirement(11, "Alpha plans production cutover on 12 October 2026 at 22:00 UTC")],
    "Q21": [requirement(12, "Beta plans its production launch on 28 November 2026 at 04:30 UTC")],
    "Q22": [requirement(12, "Beta owner: Daniel Cho")],
    "Q23": [
        requirement(13, "within 30 calendar days after the initial paid activation"),
        requirement(14, "have a 7-calendar-day refund window starting from the migration activation date"),
        requirement(14, "This exception overrides the normal 30-day Enterprise refund rule"),
    ],
    "Q24": [requirement(14, "does not apply to a customer that merely upgraded from Nova Plus to Nova Enterprise")],
    "Q25": [
        requirement(15, "Any export using AX-4 is prohibited from traversing region East-2"),
        requirement(16, "For financial exports, Orion mandates encryption mode AX-4"),
    ],
    "Q26": [requirement(15, "Encryption mode AX-3 may use East-2")],
    "Q27": [
        requirement(17, "NTP UDP/123 after a ruleset change, causing 734 seconds of drift"),
        requirement(17, "The queue pressure was real but did not cause EDG-2718")],
    "Q28": [requirement(17, "EDG-2718 is emitted when gateway clock drift exceeds 600 seconds")],
    "Q29": [
        requirement(18, "Nova Plus Severity 1 has a 2-business-hour target"),
        requirement(18, "the remaining 90 minutes resume Monday at 09:00"),
    ],
    "Q30": [requirement(18, "Enterprise Severity 1 response is measured continuously, 24x7")],
    "Q31": [
        requirement(19, "Storage billing uses the average of daily 00:00 UTC snapshots"),
        requirement(19, "has no storage overage from that noon peak alone"),
    ],
    "Q32": [requirement(20, "A credit issued on 20 August 2026 therefore uses the 180-day rule unless its memo explicitly sets an earlier expiration")],
    "Q33": [
        requirement(20, "Credits issued before 1 July 2026 remain governed by their original terms"),
        requirement(20, "unless the individual credit memo specifies an earlier date"),
    ],
    "Q34": [requirement(21, "Each subsequent F2 retry uses 90 seconds")],
    "Q35": [requirement(21, "F-2 is not a valid alias for F2")],
    "Q36": [requirement(22, "still allow a specifically marked break-glass account to use a password")],
    "Q37": [requirement(22, "Merely belonging to the Administrators group does not create this exemption")],
    "Q38": [
        requirement(23, "Recovery Point Objective (RPO) of 15 minutes"),
        requirement(23, "standard Recovery Time Objective (RTO) is 4 hours"),
    ],
    "Q42": [
        requirement(25, "Do not infer hidden state from dates, plan names, or neighboring examples"),
        requirement(25, "Enterprise tenant whose current Legacy Network Mode state is unknown"),
    ],
    "Q43": [requirement(26, "M-482 completion does not imply retention migration completion")],
    "Q44": [requirement(27, "A newer generic statement does not automatically override an older but explicitly scoped contractual exception")],
    "Q45": [requirement(27, "BILL-4.3 keys off credit issuance date")],
    "Q46": [requirement(27, "exception keys its refund window off migration activation date")],
    "Q47": [
        requirement(4, "Nova Plus includes 2,000,000 billable API calls per calendar month"),
        requirement(4, "the first 50,000 calls made through the Compliance Export API"),
        requirement(4, "This exclusion applies only to the Compliance Export API, not to the general REST API"),
    ],
    "Q48": [
        requirement(4, "Nova Plus includes 2,000,000 billable API calls per calendar month"),
        requirement(4, "the first 50,000 calls made through the Compliance Export API"),
        requirement(4, "This exclusion applies only to the Compliance Export API, not to the general REST API"),
    ],
    "Q49": [requirement(7, "On Gateway 8.4 with TLS passthrough disabled, upstream handshake failures use ERR-4817 instead")],
    "Q50": [requirement(5, "Nova Edge local requests use a 20-second local timeout; cloud synchronization uses 45 seconds")],
    "Q51": [
        requirement(8, "A separately purchased Secure Connectivity add-on can enable them"),
        requirement(8, "that add-on does not override the LNM incompatibility"),
    ],
    "Q52": [requirement(9, "After migration, the tenant receives the 90-day default unless its contract specifies a longer period")],
    "Q53": [requirement(10, "Enterprise Compliance Export jobs may run for 90 minutes")],
    "Q54": [requirement(10, "Nova Enterprise standard jobs may run for 60 minutes")],
    "Q55": [requirement(11, "Alpha change ticket: CHG-A-771")],
    "Q56": [requirement(12, "Beta change ticket: CHG-B-771")],
    "Q58": [requirement(26, '"Retention" may refer to event retention, backup retention, or legal hold')],
    "Q59": [requirement(27, "A document marked HISTORICAL or SUPERSEDED may be useful to explain old incidents")],
    "Q60": [requirement(24, "CEO approval for acquisitions above USD 10 million")],
}


# A few questions have more than one independently valid route to the answer.
# Keeping the routes separate prevents recall from demanding every duplicate
# expression of the same policy.
ALTERNATIVE_EVIDENCE_RULES: dict[str, list[list[EvidenceRequirement]]] = {
    "Q10": [
        [
            requirement(
                6,
                'An operations memo dated 3 January 2026 states "Enterprise EU export timeout = 30 seconds."',
            ),
            requirement(
                5,
                "Migration M-482 created a specific exception. Nova Enterprise tenants in EU regions that have completed M-482",
            ),
            requirement(
                5,
                "use synchronous Compliance Export requests use a 45-second application timeout",
            ),
        ]
    ],
    "Q32": [
        [
            requirement(
                20,
                "Credits issued on or after 1 July 2026 expire 180 days after issuance unless the individual credit memo specifies an earlier date",
            )
        ]
    ],
}


def normalize_text(value: str) -> str:
    """Make whitespace and case differences irrelevant during exact matching."""

    return re.sub(r"\s+", " ", value).strip().casefold()


def load_raw_dataset(path: Path) -> list[dict[str, Any]]:
    """Load raw JSON objects so source section labels remain untouched."""

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error

        record_id = record.get("id")
        if not record_id or record_id in seen_ids:
            raise ValueError(f"Missing or duplicate id at {path}:{line_number}")
        seen_ids.add(record_id)
        records.append(record)

    if not records:
        raise ValueError(f"Dataset contains no records: {path}")
    return records


def find_evidence_matches(
    requirement_item: EvidenceRequirement,
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find chunks containing one exact evidence phrase on its cited page."""

    expected_page = requirement_item.page - 1
    expected_phrase = normalize_text(requirement_item.phrase)
    return [
        chunk
        for chunk in chunks
        if chunk["metadata"].get("page") == expected_page
        and expected_phrase in normalize_text(chunk["content"])
    ]


def preview(content: str, length: int = 220) -> str:
    """Create a short one-line excerpt for manual report review."""

    compact = re.sub(r"\s+", " ", content).strip()
    return compact if len(compact) <= length else compact[: length - 3] + "..."


def map_record(record: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Map one human-authored record without using retrieval output."""

    record_id = record["id"]
    report_item: dict[str, Any] = {
        "id": record_id,
        "question": record["question"],
        "reference_sources": record.get("reference_sources", []),
        "mapped_chunk_ids": [],
        "chunks": [],
        "status": "unresolved",
        "details": "",
    }

    if not record.get("answerable", True):
        record["expected_chunk_ids"] = []
        report_item["status"] = "intentionally_unanswerable"
        report_item["details"] = "No supporting answer chunk should be labeled."
        return report_item

    rules = EVIDENCE_RULES.get(record_id)
    if not rules:
        record["expected_chunk_ids"] = []
        report_item["details"] = (
            "No complete, confident evidence rule exists within the human-cited pages."
        )
        return report_item

    cited_pages = {
        source["page"]
        for source in record.get("reference_sources", [])
        if source.get("source") == NORTHSTAR_SOURCE
    }
    alternative_rules = ALTERNATIVE_EVIDENCE_RULES.get(record_id, [])
    all_rules = rules + [
        rule
        for evidence_set in alternative_rules
        for rule in evidence_set
    ]
    invalid_rule_pages = sorted({rule.page for rule in all_rules} - cited_pages)
    if invalid_rule_pages:
        record["expected_chunk_ids"] = []
        report_item["details"] = (
            f"Evidence rule uses pages not present in human citations: {invalid_rule_pages}"
        )
        return report_item

    selected_by_id: dict[str, dict[str, Any]] = {}
    ambiguous_matches: list[dict[str, Any]] = []
    missing_phrases: list[str] = []

    for rule in rules:
        matches = find_evidence_matches(rule, chunks)
        if not matches:
            missing_phrases.append(rule.phrase)
        elif len(matches) > 1:
            ambiguous_matches.append(
                {"page": rule.page, "phrase": rule.phrase, "chunk_ids": [chunk["id"] for chunk in matches]}
            )
        else:
            selected_by_id[matches[0]["id"]] = matches[0]

    if ambiguous_matches:
        record["expected_chunk_ids"] = []
        report_item["status"] = "ambiguous"
        report_item["details"] = "One or more evidence phrases matched multiple chunks."
        report_item["ambiguous_matches"] = ambiguous_matches
        return report_item

    if missing_phrases:
        record["expected_chunk_ids"] = []
        report_item["details"] = "Missing evidence phrases in cited chunks."
        report_item["missing_phrases"] = missing_phrases
        return report_item

    selected_chunks = sorted(
        selected_by_id.values(),
        key=lambda chunk: chunk["metadata"].get("chunk_number", 0),
    )
    selected_ids = [chunk["id"] for chunk in selected_chunks]

    evidence_sets = [selected_ids]
    for alternative in alternative_rules:
        alternative_chunks: dict[str, dict[str, Any]] = {}
        for alternative_rule in alternative:
            matches = find_evidence_matches(alternative_rule, chunks)
            if len(matches) != 1:
                record["expected_chunk_ids"] = []
                report_item["status"] = (
                    "ambiguous" if len(matches) > 1 else "unresolved"
                )
                report_item["details"] = (
                    "An alternative evidence phrase did not match exactly one chunk."
                )
                report_item["alternative_phrase"] = alternative_rule.phrase
                report_item["alternative_matches"] = [
                    chunk["id"] for chunk in matches
                ]
                return report_item
            alternative_chunks[matches[0]["id"]] = matches[0]

        ordered_alternative = sorted(
            alternative_chunks.values(),
            key=lambda chunk: chunk["metadata"].get("chunk_number", 0),
        )
        alternative_ids = [chunk["id"] for chunk in ordered_alternative]
        if alternative_ids not in evidence_sets:
            evidence_sets.append(alternative_ids)

    record["expected_chunk_ids"] = selected_ids
    if alternative_rules:
        record["expected_evidence_sets"] = evidence_sets
    report_item["mapped_chunk_ids"] = selected_ids
    if alternative_rules:
        report_item["expected_evidence_sets"] = evidence_sets
    report_item["chunks"] = [
        {
            "chunk_id": chunk["id"],
            "document_id": chunk["metadata"].get("document_id"),
            "page": chunk["metadata"].get("page", 0) + 1,
            "chunk_number": chunk["metadata"].get("chunk_number"),
            "content_preview": preview(chunk["content"]),
        }
        for chunk in selected_chunks
    ]
    report_item["status"] = "multi_chunk" if len(selected_ids) > 1 else "mapped"
    report_item["details"] = "All explicit evidence phrases matched exactly one cited chunk."
    return report_item


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write generated mappings while leaving the human source file unchanged."""

    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    path.write_text(content, encoding="utf-8")


def run(dataset_path: Path, output_path: Path, report_path: Path) -> dict[str, int]:
    """Load database chunks, map all records, validate IDs, and write outputs."""

    records = load_raw_dataset(dataset_path)
    chunks = get_document_chunks(NORTHSTAR_SOURCE)
    if len(chunks) != 58:
        raise RuntimeError(
            f"Expected 58 chunks for {NORTHSTAR_SOURCE}, found {len(chunks)}"
        )

    existing_ids = {chunk["id"] for chunk in chunks}
    report_items = [map_record(record, chunks) for record in records]
    mapped_ids = {
        chunk_id
        for record in records
        for chunk_id in record.get("expected_chunk_ids", [])
    }
    mapped_ids.update(
        chunk_id
        for record in records
        for evidence_set in record.get("expected_evidence_sets", [])
        for chunk_id in evidence_set
    )
    missing_ids = sorted(mapped_ids - existing_ids)
    if missing_ids:
        raise RuntimeError(f"Mapped chunk IDs do not exist: {missing_ids}")

    status_order = (
        "mapped",
        "multi_chunk",
        "intentionally_unanswerable",
        "unresolved",
        "ambiguous",
    )
    summary = {
        "total_questions": len(records),
        **{
            status: sum(item["status"] == status for item in report_items)
            for status in status_order
        },
    }
    report = {
        "source_dataset": str(dataset_path.resolve()),
        "mapped_dataset": str(output_path.resolve()),
        "database_source": NORTHSTAR_SOURCE,
        "database_chunk_count": len(chunks),
        "summary": summary,
        "questions": report_items,
    }

    write_jsonl(output_path, records)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    """Read optional paths while keeping the requested command argument-free."""

    parser = argparse.ArgumentParser(description="Map Northstar ground truth to stored chunks")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    """Run with ``python -m evals.map_ground_truth``."""

    args = parse_args()
    summary = run(args.dataset, args.output, args.report)
    print(f"Total questions: {summary['total_questions']}")
    print(f"Mapped: {summary['mapped']}")
    print(f"Multi-chunk: {summary['multi_chunk']}")
    print(f"Unanswerable: {summary['intentionally_unanswerable']}")
    print(f"Unresolved: {summary['unresolved']}")
    print(f"Ambiguous: {summary['ambiguous']}")


if __name__ == "__main__":
    main()
