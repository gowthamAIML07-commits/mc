"""Dataset Registry and License Verification Engine."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
from data.registry.models import DatasetEntry, DatasetRegistry, DatasetRole, LicenseStatus


class DatasetLicenseVerifier:
    """Validator and governance verifier for datasets and licensing compliance."""

    PERMISSIVE_LICENSES = {
        "CC BY 4.0",
        "CC BY-SA 4.0",
        "MIT License",
        "Apache 2.0",
        "Open Access Public Health Data (NLM)",
        "Public Domain",
    }

    RESTRICTED_LICENSES = {
        "UMLS Metathesaurus License (NLM/NIH)",
    }

    RESEARCH_ONLY_LICENSES = {
        "Academic / Research Open Access",
    }

    def __init__(self, manifest_path: Optional[Path] = None):
        if manifest_path is None:
            root_dir = Path(__file__).resolve().parent.parent.parent
            manifest_path = root_dir / "data" / "registry" / "datasets.yaml"
        self.manifest_path = Path(manifest_path)
        self.registry: Optional[DatasetRegistry] = None
        self._load_registry()

    def _load_registry(self) -> None:
        """Load and parse the registry YAML."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Registry manifest not found at: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        self.registry = DatasetRegistry(**raw_data)

    def get_dataset(self, key_or_id: str) -> Optional[DatasetEntry]:
        """Retrieve dataset entry by registry key or ID (e.g. 'DS-1')."""
        if not self.registry:
            return None

        # Check key direct lookup
        if key_or_id in self.registry.datasets:
            return self.registry.datasets[key_or_id]

        # Check by dataset ID
        for entry in self.registry.datasets.values():
            if entry.id.lower() == key_or_id.lower():
                return entry

        return None

    def evaluate_license(self, dataset: DatasetEntry) -> LicenseStatus:
        """Classify the license compliance tier of a dataset entry."""
        lic = dataset.license.strip()
        if lic in self.PERMISSIVE_LICENSES:
            return LicenseStatus.VERIFIED_PERMISSIVE
        elif lic in self.RESTRICTED_LICENSES:
            return LicenseStatus.VERIFIED_RESTRICTED
        elif lic in self.RESEARCH_ONLY_LICENSES:
            return LicenseStatus.VERIFIED_RESEARCH_ONLY
        else:
            return LicenseStatus.UNRESOLVED

    def can_download(
        self,
        key_or_id: str,
        allow_restricted_uts: bool = False,
        allow_non_commercial: bool = True
    ) -> Tuple[bool, str]:
        """Safety guard: determines if dataset is permitted for automated download."""
        entry = self.get_dataset(key_or_id)
        if not entry:
            return False, f"Dataset '{key_or_id}' not found in registry manifest."

        status = self.evaluate_license(entry)

        if status == LicenseStatus.UNRESOLVED:
            return False, f"BLOCKED: Dataset '{entry.name}' has unresolved license '{entry.license}'. Download prohibited."

        if status == LicenseStatus.VERIFIED_RESTRICTED:
            if entry.uts_license_required and not allow_restricted_uts:
                return False, (
                    f"MANUAL REVIEW REQUIRED: Dataset '{entry.name}' requires UMLS UTS license agreement. "
                    "Cannot be automatically downloaded without verified UTS credentials."
                )

        if status == LicenseStatus.VERIFIED_RESEARCH_ONLY:
            if not allow_non_commercial:
                return False, f"BLOCKED: Dataset '{entry.name}' is restricted to non-commercial academic research."

        return True, f"PERMITTED: License '{entry.license}' validated for task '{entry.task}'."

    def verify_all(self) -> Dict[str, dict]:
        """Verify all registered datasets and return comprehensive audit results."""
        if not self.registry:
            raise RuntimeError("Registry not loaded.")

        results = {}
        for key, entry in self.registry.datasets.items():
            status = self.evaluate_license(entry)
            downloadable, reason = self.can_download(key, allow_restricted_uts=False)
            
            results[key] = {
                "id": entry.id,
                "name": entry.name,
                "task": entry.task,
                "role": entry.role,
                "source_url": entry.source_url,
                "version": entry.version,
                "license": entry.license,
                "license_status": status.value,
                "redistribution_allowed": entry.redistribution_allowed,
                "commercial_use_allowed": entry.commercial_use_allowed,
                "limitations": entry.limitations,
                "download_permitted_automatically": downloadable,
                "download_policy_notes": reason,
                "manual_review_required": entry.manual_review_required or (status != LicenseStatus.VERIFIED_PERMISSIVE)
            }
        return results

    def get_datasets_by_role(self, role: Union[DatasetRole, str]) -> List[DatasetEntry]:
        """Filter registered datasets by role (training, knowledge_base, etc.)."""
        if not self.registry:
            return []
        role_val = role.value if isinstance(role, DatasetRole) else str(role)
        return [ds for ds in self.registry.datasets.values() if ds.role == role_val]
