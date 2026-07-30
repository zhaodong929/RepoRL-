"""Defensive JUnit XML parsing independent of console output."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable

from reporl.verifier.models import JUnitReport, JUnitTestCase, TestCaseStatus

MAX_JUNIT_BYTES = 10_000_000


class JUnitParseError(ValueError):
    """JUnit evidence is malformed, unsafe, empty, or ambiguous."""


def parse_junit_xml(
    xml: str | bytes,
    *,
    expected_test_ids: tuple[str, ...] = (),
) -> JUnitReport:
    payload = xml.encode("utf-8") if isinstance(xml, str) else xml
    if not payload or len(payload) > MAX_JUNIT_BYTES:
        raise JUnitParseError("JUnit XML is empty or exceeds the size limit")
    lowered = payload.lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        raise JUnitParseError("DTD and entity declarations are not allowed in JUnit XML")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as error:
        raise JUnitParseError("malformed JUnit XML") from error
    if _local_name(root.tag) not in {"testsuite", "testsuites"}:
        raise JUnitParseError("JUnit root must be testsuite or testsuites")

    cases: list[JUnitTestCase] = []
    seen_ids: set[str] = set()
    for element in _iter_testcases(root):
        name = element.attrib.get("name", "").strip()
        classname = element.attrib.get("classname", "").strip()
        file_name = element.attrib.get("file", "").strip()
        if not name:
            raise JUnitParseError("JUnit testcase is missing a name")
        prefix = classname or file_name
        test_id = f"{prefix}::{name}" if prefix else name
        if test_id in seen_ids:
            raise JUnitParseError(f"duplicate JUnit testcase ID: {test_id}")
        seen_ids.add(test_id)
        status, message = _case_status(element)
        duration = _parse_duration(element.attrib.get("time", "0"))
        cases.append(
            JUnitTestCase(
                test_id=test_id,
                classname=classname,
                name=name,
                status=status,
                duration_seconds=duration,
                message=message[:4_000],
            )
        )
    if not cases:
        raise JUnitParseError("JUnit XML contains no testcases")

    actual_ids = set(seen_ids)
    expected_ids = set(expected_test_ids)
    return JUnitReport(
        cases=tuple(cases),
        expected_test_ids=expected_test_ids,
        missing_test_ids=tuple(sorted(expected_ids - actual_ids)),
        unexpected_test_ids=tuple(sorted(actual_ids - expected_ids)) if expected_ids else (),
    )


def _iter_testcases(root: ET.Element) -> Iterable[ET.Element]:
    for element in root.iter():
        if _local_name(element.tag) == "testcase":
            yield element


def _case_status(element: ET.Element) -> tuple[TestCaseStatus, str]:
    outcome: tuple[TestCaseStatus, str] = (TestCaseStatus.PASSED, "")
    seen_outcome = False
    for child in element:
        name = _local_name(child.tag)
        if name not in {"failure", "error", "skipped"}:
            continue
        if seen_outcome:
            raise JUnitParseError("JUnit testcase has multiple outcome elements")
        seen_outcome = True
        status = {
            "failure": TestCaseStatus.FAILED,
            "error": TestCaseStatus.ERROR,
            "skipped": TestCaseStatus.SKIPPED,
        }[name]
        message = child.attrib.get("message", "") or (child.text or "")
        outcome = status, message.strip()
    return outcome


def _parse_duration(raw: str) -> float:
    try:
        duration = float(raw)
    except ValueError as error:
        raise JUnitParseError("JUnit testcase has an invalid duration") from error
    if duration < 0 or duration == float("inf") or duration != duration:
        raise JUnitParseError("JUnit testcase duration must be finite and non-negative")
    return duration


def _local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]
