from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


FORBIDDEN_TEST_PATH_MARKERS = (
    "test_images",
    "testing_images",
    "/test_images/",
    "/testing_images/",
)


@dataclass
class LeakageReport:
    run_name: str
    train_split_path: str = ""
    validation_split_path: str = ""
    test_eeg_path: str = ""
    feature_caches_used: list[str] = field(default_factory=list)
    accessed_paths: list[str] = field(default_factory=list)
    used_test_image_path: bool = False
    used_test_label_or_class_prompt: bool = False
    memory_bank_source: str = "train-only"
    notes: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            f"run_name: {self.run_name}",
            f"train split path: {self.train_split_path}",
            f"validation split path: {self.validation_split_path}",
            f"test EEG path: {self.test_eeg_path}",
            "feature caches used:",
        ]
        lines.extend(f"- {item}" for item in self.feature_caches_used)
        lines.extend(
            [
                f"whether any test image path was accessed: {self.used_test_image_path}",
                f"whether any test label/class prompt was used: {self.used_test_label_or_class_prompt}",
                f"memory bank source: {self.memory_bank_source}",
                "notes:",
            ]
        )
        lines.extend(f"- {item}" for item in self.notes)
        return "\n".join(lines) + "\n"


class LeakageGuard:
    """Central guardrail for train/validation-only tuning and test-time inference."""

    def __init__(self, config, run_name: str = "run") -> None:
        self.config = config
        self.report = LeakageReport(run_name=run_name)
        self.assert_static_config()

    def assert_static_config(self) -> None:
        inference = _get(self.config, "inference", {})
        assert inference.get("use_test_candidate_bank", False) is False
        assert inference.get("use_test_gt_img2img", False) is False
        assert inference.get("use_test_label_prompt", False) is False
        if "test" in str(inference.get("prototype_bank_name", "")).lower():
            raise AssertionError("prototype_bank_name must not contain 'test'")

    def register_split_paths(self, train: str | Path, val: str | Path, test_eeg: str | Path) -> None:
        self.report.train_split_path = str(train)
        self.report.validation_split_path = str(val)
        self.report.test_eeg_path = str(test_eeg)

    def register_feature_cache(self, path: str | Path) -> None:
        path_str = str(path)
        self.report.feature_caches_used.append(path_str)
        self.assert_no_test_image_path(path_str, context="feature cache")

    def assert_train_memory_bank_paths(self, paths: Iterable[str | Path]) -> None:
        path_list = [str(path) for path in paths]
        assert "test_images" not in " ".join(path_list)
        for path in path_list:
            self.assert_no_test_image_path(path, context="train memory bank")
        self.report.memory_bank_source = "train-only"

    def assert_no_test_image_path(self, path: str | Path, context: str = "") -> None:
        path_str = str(path)
        normalized = path_str.replace("\\", "/").lower()
        self.report.accessed_paths.append(path_str)
        if any(marker in normalized for marker in FORBIDDEN_TEST_PATH_MARKERS):
            self.report.used_test_image_path = True
            raise AssertionError(f"Forbidden test image path in {context}: {path_str}")

    def assert_prompt_is_neutral(self, prompt: str, forbidden_labels: Iterable[str] | None = None) -> None:
        prompt_norm = prompt.lower()
        if forbidden_labels:
            for label in forbidden_labels:
                label_norm = str(label).replace("_", " ").lower()
                if label_norm and label_norm in prompt_norm:
                    self.report.used_test_label_or_class_prompt = True
                    raise AssertionError(f"Prompt contains test label/class name: {label}")
        allowed = {
            "a natural image, high quality, realistic",
            "",
        }
        if prompt_norm not in allowed:
            self.report.notes.append(f"non-default neutral prompt used: {prompt}")

    def write_report(self, output_dir: str | Path) -> Path:
        path = Path(output_dir) / "leakage_report.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.report.to_text(), encoding="utf-8")
        return path


def _get(config, key: str, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)
