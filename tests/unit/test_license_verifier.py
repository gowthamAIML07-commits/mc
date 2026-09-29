"""Unit tests for the License Verification and Download Safety Guard."""
import pytest
from data.registry.verifier import DatasetLicenseVerifier
from data.registry.models import DatasetEntry, LicenseStatus


@pytest.fixture
def verifier():
    return DatasetLicenseVerifier()


def test_evaluate_license_permissive(verifier):
    """Verify permissive licenses are classified as VERIFIED_PERMISSIVE."""
    ds1 = verifier.get_dataset("DS-1")
    assert verifier.evaluate_license(ds1) == LicenseStatus.VERIFIED_PERMISSIVE

    ds2 = verifier.get_dataset("DS-2")
    assert verifier.evaluate_license(ds2) == LicenseStatus.VERIFIED_PERMISSIVE

    ds4 = verifier.get_dataset("DS-4")
    assert verifier.evaluate_license(ds4) == LicenseStatus.VERIFIED_PERMISSIVE

    ds6 = verifier.get_dataset("DS-6")
    assert verifier.evaluate_license(ds6) == LicenseStatus.VERIFIED_PERMISSIVE


def test_evaluate_license_restricted_and_research(verifier):
    """Verify restricted UMLS and research-only licenses are properly classified."""
    rxnorm = verifier.get_dataset("DS-3")
    assert verifier.evaluate_license(rxnorm) == LicenseStatus.VERIFIED_RESTRICTED

    funsd = verifier.get_dataset("DS-5")
    assert verifier.evaluate_license(funsd) == LicenseStatus.VERIFIED_RESEARCH_ONLY


def test_download_guard_permissive_datasets(verifier):
    """Verify automated download is permitted for open permissive datasets."""
    allowed, msg = verifier.can_download("DS-1")
    assert allowed is True
    assert "PERMITTED" in msg

    allowed, msg = verifier.can_download("DS-2")
    assert allowed is True

    allowed, msg = verifier.can_download("DS-4")
    assert allowed is True

    allowed, msg = verifier.can_download("DS-6")
    assert allowed is True


def test_download_guard_blocks_unauthorized_uts_license(verifier):
    """Verify automated download for RxNorm is blocked unless explicit UTS auth is provided."""
    allowed, msg = verifier.can_download("DS-3", allow_restricted_uts=False)
    assert allowed is False
    assert "MANUAL REVIEW REQUIRED" in msg

    # When explicit UTS license acceptance is flagged
    allowed, msg = verifier.can_download("DS-3", allow_restricted_uts=True)
    assert allowed is True


def test_download_guard_blocks_non_commercial_in_commercial_mode(verifier):
    """Verify FUNSD is blocked if non-commercial usage is disallowed."""
    allowed, msg = verifier.can_download("DS-5", allow_non_commercial=False)
    assert allowed is False
    assert "BLOCKED" in msg

    # Allowed for academic/research evaluation
    allowed, msg = verifier.can_download("DS-5", allow_non_commercial=True)
    assert allowed is True


def test_download_guard_blocks_unresolved_license(verifier):
    """Verify fake/unresolved dataset licenses are strictly prohibited from download."""
    unresolved_entry = DatasetEntry(
        id="DS-99",
        name="Mystery Prescription Dataset",
        task="testing",
        role="training",
        source_url="https://example.com/data",
        version="v1.0",
        license="Unknown Proprietary License",
        redistribution_allowed=False,
        commercial_use_allowed=False,
        limitations="Unclear licensing terms",
        local_path="data/raw/mystery",
        processed_path="data/processed/mystery",
        format="csv"
    )
    # Temporarily attach to registry for test
    verifier.registry.datasets["mystery"] = unresolved_entry

    assert verifier.evaluate_license(unresolved_entry) == LicenseStatus.UNRESOLVED
    allowed, msg = verifier.can_download("mystery")
    assert allowed is False
    assert "unresolved license" in msg


def test_verify_all_audit_report(verifier):
    """Verify audit report generates comprehensive verification details."""
    report = verifier.verify_all()
    assert len(report) >= 6

    # Verify RxNorm requires manual review
    assert report["rxnorm"]["manual_review_required"] is True
    assert report["rxnorm"]["license_status"] == "verified_restricted"

    # Verify FUNSD requires manual review
    assert report["funsd"]["manual_review_required"] is True
    assert report["funsd"]["license_status"] == "verified_research_only"

    # Verify Indian Rx OCR does not require manual review
    assert report["indian_medical_prescription_ocr"]["manual_review_required"] is False
    assert report["indian_medical_prescription_ocr"]["license_status"] == "verified_permissive"
