"""BNF-to-dm+d medicine price lookup.

The module deliberately has no Streamlit dependency so it can be used from a
command line job, tests, or a web interface.  Prices in the supplied dm+d XML
are stored in pence and are converted to GBP on output.
"""
from __future__ import annotations

import csv
import gzip
import io
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Iterable

from openpyxl import load_workbook


OUTPUT_COLUMNS = [
    "input_row", "input_bnf_code", "bnf_level",
    "message", "medicine_name", "price", "pack_size", "dmd_type",
    "price_basis", "snomed_dmd_id", "price_effective_date",
    "discontinued_date",
]


@dataclass(frozen=True)
class Concept:
    concept_type: str
    concept_id: str
    bnf_name: str
    pack: str


def _text(element: ET.Element, tag: str) -> str:
    child = element.find(tag)
    return (child.text or "").strip() if child is not None else ""


def _normalise_code(value: object) -> str:
    raw = str(value or "").upper().strip()
    # Accept either a bare BNF code or a display value such as
    # "Ursodeoxycholic acid 0109010U0".
    embedded_codes = re.findall(
        r"(?<![A-Z0-9])([0-9]{7}[A-Z][0-9](?:[A-Z]{2}(?:[A-Z]{4})?)?)(?![A-Z0-9])",
        raw,
    )
    if embedded_codes:
        return embedded_codes[-1]
    return re.sub(r"[^A-Z0-9]", "", raw)


def _iso_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _select_current(prices: list[dict], as_of: date) -> dict | None:
    """Return the newest record in force at as_of; otherwise the newest known."""
    if not prices:
        return None
    dated = [p for p in prices if p["effective"] and p["effective"] <= as_of]
    candidates = dated or prices
    return max(candidates, key=lambda p: p["effective"] or date.min)


class DmdPriceLookup:
    """Loads the provided BNF mapping workbook and the AMPP/VMPP dm+d files."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.mapping_source = self._required("BNF Snomed Mapping data *.xlsx")
        self.ampp_source = self._required("f_ampp2_*.xml*")
        self.vmpp_source = self._required("f_vmpp2_*.xml")
        self.mapping: dict[str, list[Concept]] = defaultdict(list)
        self.ampps: dict[str, dict] = {}
        self.vmpps: dict[str, dict] = {}
        self.ampp_by_amp: dict[str, set[str]] = defaultdict(set)
        self.vmpp_by_vmp: dict[str, set[str]] = defaultdict(set)
        self.priced_ampp_by_amp: dict[str, set[str]] = defaultdict(set)
        self.priced_vmpp_by_vmp: dict[str, set[str]] = defaultdict(set)
        self.priced_ampp_ids: set[str] = set()
        self.priced_vmpp_ids: set[str] = set()
        self.known_ampp_ids: set[str] = set()
        self.known_vmpp_ids: set[str] = set()
        self._load()

    def _required(self, pattern: str) -> Path:
        matches = sorted(self.data_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"Required data file not found: {pattern}. Put the supplied dm+d "
                f"files and BNF mapping workbook in {self.data_dir}."
            )
        return matches[-1]

    @staticmethod
    def _iterparse_xml(source: Path):
        """Iterate an XML extract, transparently supporting gzip-compressed files."""
        if source.suffix == ".gz":
            with gzip.open(source, "rb") as handle:
                yield from ET.iterparse(handle, events=("end",))
        else:
            yield from ET.iterparse(source, events=("end",))

    def _load(self) -> None:
        self._load_mapping()
        self._load_ampps()
        self._load_vmpps()
        self._build_price_indexes()

    def _build_price_indexes(self) -> None:
        """Keep expansion indexes limited to packs with a potentially usable price.

        The price tables follow the product tables in the supplied XML, so a
        one-pass parse cannot reject unpriced products immediately. Building
        these indexes immediately afterwards prevents unpriced AMPP/VMPP
        records from being expanded for chemical/product-level searches.
        """
        self.known_ampp_ids = set(self.ampps)
        self.known_vmpp_ids = set(self.vmpps)
        for appid, record in self.ampps.items():
            if any(price.get("basis") not in {"0002", "0003", "0004"} for price in record["prices"]):
                self.priced_ampp_ids.add(appid)
                self.priced_ampp_by_amp[record["amp_id"]].add(appid)
        for vppid, record in self.vmpps.items():
            if record["prices"]:
                self.priced_vmpp_ids.add(vppid)
                self.priced_vmpp_by_vmp[record["vmp_id"]].add(vppid)
        # The complete product tables are large. Once the relational joins and
        # price-only indexes exist, unpriced records cannot contribute to the
        # requested output and can be released from memory.
        self.ampps = {key: value for key, value in self.ampps.items() if key in self.priced_ampp_ids}
        self.vmpps = {key: value for key, value in self.vmpps.items() if key in self.priced_vmpp_ids}
        self.ampp_by_amp.clear()
        self.vmpp_by_vmp.clear()

    def _load_mapping(self) -> None:
        workbook = load_workbook(self.mapping_source, read_only=True, data_only=True)
        sheet = workbook.active
        headers = [str(v or "").strip() for v in next(sheet.values)]
        required = {"BNF Code", "BNF Name", "SNOMED Code", "VMP / VMPP/ AMP / AMPP", "Pack"}
        missing = required - set(headers)
        if missing:
            raise ValueError(f"Mapping workbook is missing columns: {', '.join(sorted(missing))}")
        indexes = {header: headers.index(header) for header in required}
        for row in sheet.values:
            code = _normalise_code(row[indexes["BNF Code"]])
            concept_id = str(row[indexes["SNOMED Code"]] or "").strip()
            concept_type = str(row[indexes["VMP / VMPP/ AMP / AMPP"]] or "").strip().upper()
            if not code or not concept_id or concept_type not in {"VMP", "VMPP", "AMP", "AMPP"}:
                continue
            concept = Concept(
                concept_type, concept_id, str(row[indexes["BNF Name"]] or "").strip(),
                str(row[indexes["Pack"]] or "").strip(),
            )
            if concept not in self.mapping[code]:
                self.mapping[code].append(concept)
        workbook.close()

    def _load_ampps(self) -> None:
        for _, element in self._iterparse_xml(self.ampp_source):
            if element.tag == "AMPP":
                appid = _text(element, "APPID")
                self.ampps[appid] = {
                    "id": appid, "name": _text(element, "NM"), "vmpp_id": _text(element, "VPPID"),
                    "amp_id": _text(element, "APID"), "sub_pack": _text(element, "SUBP"),
                    "invalid": _text(element, "INVALID"), "discontinued": _text(element, "DISCDT"),
                    "prices": [],
                }
                if _text(element, "APID"):
                    self.ampp_by_amp[_text(element, "APID")].add(appid)
                element.clear()
            # PRICE_INFO is a separate, later table in the dm+d XML—not a
            # child of AMPP. Its APPID is the relational join key.
            elif element.tag == "PRICE_INFO":
                appid, raw = _text(element, "APPID"), _text(element, "PRICE")
                if appid in self.ampps and raw:
                    self.ampps[appid]["prices"].append({
                        "pence": raw, "effective": _iso_date(_text(element, "PRICEDT")),
                        "basis": _text(element, "PRICE_BASISCD"),
                    })
                element.clear()

    def _load_vmpps(self) -> None:
        for _, element in ET.iterparse(self.vmpp_source, events=("end",)):
            if element.tag == "VMPP":
                vppid = _text(element, "VPPID")
                self.vmpps[vppid] = {
                    "id": vppid, "name": _text(element, "NM"), "vmp_id": _text(element, "VPID"),
                    "quantity": " ".join(filter(None, [_text(element, "QTYVAL"), _text(element, "QTY_UOMCD")])),
                    "invalid": _text(element, "INVALID"), "prices": [],
                }
                if _text(element, "VPID"):
                    self.vmpp_by_vmp[_text(element, "VPID")].add(vppid)
                element.clear()
            # DTINFO is likewise a separate relational table linked by VPPID.
            elif element.tag == "DTINFO":
                vppid, raw = _text(element, "VPPID"), _text(element, "PRICE")
                if vppid in self.vmpps and raw:
                    self.vmpps[vppid]["prices"].append({"pence": raw, "effective": _iso_date(_text(element, "DT"))})
                element.clear()

    @staticmethod
    def _level(code: str) -> str | None:
        # Standard BNF code lengths used by the supplied mapping:
        # 9 characters = chemical substance, 11 = product, 15 = presentation.
        return {
            9: "Chemical",
            11: "Product",
            15: "Presentation",
        }.get(len(code))

    def _mapped_concepts(self, code: str) -> list[Concept]:
        if len(code) == 9:
            # Chemical codes are the common prefix of their product and
            # presentation mapping rows, so collect all descendants.
            concepts = [c for key, values in self.mapping.items() if key.startswith(code) for c in values]
        else:
            concepts = list(self.mapping.get(code, []))
        return list(dict.fromkeys(concepts))

    def _expand(self, concept: Concept) -> Iterable[tuple[str, str, str, str]]:
        """Yield (type, id, BNF name, mapping pack) at requested pack granularity."""
        if concept.concept_type == "AMPP":
            if concept.concept_id in self.priced_ampp_ids or concept.concept_id not in self.known_ampp_ids:
                yield "AMPP", concept.concept_id, concept.bnf_name, concept.pack
        elif concept.concept_type == "VMPP":
            if concept.concept_id in self.priced_vmpp_ids or concept.concept_id not in self.known_vmpp_ids:
                yield "VMPP", concept.concept_id, concept.bnf_name, concept.pack
        elif concept.concept_type == "AMP":
            for appid in self.priced_ampp_by_amp.get(concept.concept_id, set()):
                yield "AMPP", appid, concept.bnf_name, concept.pack
        elif concept.concept_type == "VMP":
            for vppid in self.priced_vmpp_by_vmp.get(concept.concept_id, set()):
                yield "VMPP", vppid, concept.bnf_name, concept.pack

    def lookup_code(self, raw_code: object, input_row: int, as_of: date | None = None) -> list[dict]:
        as_of = as_of or date.today()
        code = _normalise_code(raw_code)
        level = self._level(code)
        base = {"input_row": input_row, "input_bnf_code": str(raw_code or ""), "normalised_bnf_code": code,
                "bnf_level": level or "Unsupported"}
        if not code:
            return [base | {"status": "invalid_bnf_code", "message": "Blank BNF code."}]
        if not level:
            message = (
                "BNF code is less specific than Chemical level."
                if len(code) < 9
                else (
                    "BNF code is not a supported 9-character Chemical, "
                    "11-character Product, or 15-character Presentation code."
                )
            )
            return [base | {"status": "unsupported_bnf_code", "message": message}]
        concepts = self._mapped_concepts(code)
        if not concepts:
            return [base | {"status": "missing_mapping", "message": "No BNF-to-SNOMED mapping found in the supplied workbook."}]
        # A BNF code can map to the same pack through several levels in the
        # workbook (for example VMP and VMPP). Output one row per actual pack.
        resolved_by_pack = {}
        for item in (item for concept in concepts for item in self._expand(concept)):
            key = (item[0], item[1])
            # Prefer a direct mapping row that provides a human-readable pack
            # over an expansion from VMP/AMP with no pack information.
            if key not in resolved_by_pack or (not resolved_by_pack[key][3] and item[3]):
                resolved_by_pack[key] = item
        resolved = list(resolved_by_pack.values())
        if not resolved:
            return [base | {"status": "missing_product_mapping", "message": "Mapped concept has no AMPP or VMPP in the supplied dm+d extract."}]
        ambiguous = len(resolved) > 1
        rows = []
        for concept_type, concept_id, bnf_name, mapping_pack in resolved:
            record = self.ampps.get(concept_id) if concept_type == "AMPP" else self.vmpps.get(concept_id)
            if not record:
                rows.append(base | {"status": "missing_product", "message": "Mapped concept is absent from the supplied dm+d product extract.",
                                    "bnf_name": bnf_name, "dmd_type": concept_type, "snomed_dmd_id": concept_id})
                continue
            price = _select_current(record["prices"], as_of)
            if concept_type == "AMPP":
                pack = record["sub_pack"] or mapping_pack or self.vmpps.get(record["vmpp_id"], {}).get("quantity", "")
                price_basis = {"0001": "NHS Indicative Price", "0002": "No Price Available", "0003": "No Price product centrally funded", "0004": "No Price priced when manufactured"}.get((price or {}).get("basis", ""), "")
                discontinued = record["discontinued"]
            else:
                pack, price_basis, discontinued = mapping_pack or record["quantity"], "Drug Tariff", ""
            has_price = bool(price) and not (
                concept_type == "AMPP" and (price or {}).get("basis") in {"0002", "0003", "0004"}
            )
            # The caller has explicitly requested that unpriced items are not
            # reported at all; skip them rather than emitting missing_price.
            if not has_price:
                continue
            amount = str((Decimal(price["pence"]) / Decimal("100")).quantize(Decimal("0.01")))
            status, message = ("ambiguous_match" if ambiguous else "matched"), ("Multiple AMPP/VMPP matches returned." if ambiguous else "")
            rows.append(base | {"status": status, "message": message, "medicine_name": record["name"],
                                "dmd_type": concept_type, "snomed_dmd_id": concept_id, "pack_size": pack,
                                "price": amount,
                                "price_basis": price_basis, "price_effective_date": (price or {}).get("effective", "") or "",
                                "discontinued_date": discontinued})
        return rows

    def lookup_codes(self, codes: Iterable[object], as_of: date | None = None) -> list[dict]:
        return [row for i, code in enumerate(codes, start=2) for row in self.lookup_code(code, i, as_of)]


def read_codes_csv(contents: bytes | str) -> list[str]:
    text = contents.decode("utf-8-sig") if isinstance(contents, bytes) else contents
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV must contain a header and at least one BNF code column.")
    preferred = next((h for h in reader.fieldnames if _normalise_code(h) in {"BNFCODE", "BNFCODES"}), reader.fieldnames[0])
    return [row.get(preferred, "") for row in reader]


def write_results_csv(rows: Iterable[dict], output: str | Path) -> None:
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
