from __future__ import annotations

import pytest

from reporl.verifier.junit import JUnitParseError, parse_junit_xml
from reporl.verifier.models import TestCaseStatus as CaseStatus

JUNIT = b"""<?xml version="1.0"?>
<testsuites><testsuite name="pytest">
  <testcase classname="tests.test_math" name="test_add" time="0.01" />
  <testcase classname="tests.test_math" name="test_sub" time="0.02">
    <failure message="wrong result">trace</failure>
  </testcase>
</testsuite></testsuites>
"""


def test_junit_parser_uses_testcase_outcomes_not_declared_counters() -> None:
    report = parse_junit_xml(JUNIT)

    assert report.total == 2
    assert report.passed == 1
    assert report.failed == 1
    assert report.cases[1].status == CaseStatus.FAILED
    assert not report.all_passed


def test_junit_parser_requires_exact_expected_ids() -> None:
    report = parse_junit_xml(
        JUNIT,
        expected_test_ids=("tests.test_math::test_add", "tests.test_math::test_missing"),
    )

    assert report.missing_test_ids == ("tests.test_math::test_missing",)
    assert report.unexpected_test_ids == ("tests.test_math::test_sub",)
    assert not report.all_passed


@pytest.mark.parametrize(
    "xml, message",
    [
        (b"<not-junit />", "root"),
        (b"<testsuite />", "no testcases"),
        (
            b"<testsuite><testcase name='x'/><testcase name='x'/></testsuite>",
            "duplicate",
        ),
        (
            b"<!DOCTYPE x [<!ENTITY boom 'x'>]><testsuite><testcase name='x'/></testsuite>",
            "DTD",
        ),
    ],
)
def test_junit_parser_rejects_ambiguous_or_unsafe_xml(xml: bytes, message: str) -> None:
    with pytest.raises(JUnitParseError, match=message):
        parse_junit_xml(xml)
