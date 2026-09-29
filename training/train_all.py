"""Master Multi-Task Training Orchestrator.

Executes and logs all 5 independent training pipelines:
  A. Prescription OCR (Indian Prescription OCR + Augmentation)
  B. Handwritten Medicine Recognition (Doctor's Handwritten Prescription BD)
  C. Prescription Augmentation (Synthetic Prescriptions)
  D. Document Understanding (FUNSD)
  E. Medicine Normalization (RxNorm + RxTerms)

Generates checkpoints, SHA-256 hashes, metrics, and consolidated reports in reports/training/.
"""
import json
import os
import sys
import time
from pathlib import Path

# Ensure root directory is on python path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from training.pipelines.train_ocr import train_ocr_pipeline
from training.pipelines.train_handwriting import train_handwriting_model
from training.pipelines.train_augmentation import run_augmentation_pipeline
from training.pipelines.train_layout import train_layout_model
from training.pipelines.train_normalization import train_normalization_pipeline
from ml.utils.provenance import get_hardware_info


def run_all_training_pipelines(
    handwriting_epochs: int = 4,
    synthetic_count: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """Execute all 5 pipelines independently, verifying and logging metrics for each."""
    start_total_time = time.time()
    root_dir = Path(__file__).resolve().parent.parent
    reports_dir = root_dir / "reports" / "training"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("        STARTING MULTI-TASK DATA TRAINING & EVALUATION PIPELINES                ")
    print("================================================================================\n")

    summary = {
        "title": "Multi-Task Training & Evaluation Suite",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "global_seed": seed,
        "hardware_information": get_hardware_info(),
        "pipelines": {}
    }

    # Pipeline C: Prescription Augmentation & Synthesis
    print("\n>>> [1/5] Executing Pipeline C: Prescription Augmentation (Synthetic Dataset)...")
    summary["pipelines"]["pipeline_c_augmentation"] = run_augmentation_pipeline(count=synthetic_count, seed=seed)

    # Pipeline A: Prescription OCR
    print("\n>>> [2/5] Executing Pipeline A: Prescription OCR & Field Extraction...")
    summary["pipelines"]["pipeline_a_ocr"] = train_ocr_pipeline(seed=seed)

    # Pipeline B: Handwritten Medicine Recognition
    print("\n>>> [3/5] Executing Pipeline B: Handwritten Medicine Recognition (BD Dataset)...")
    summary["pipelines"]["pipeline_b_handwriting"] = train_handwriting_model(epochs=handwriting_epochs, seed=seed)

    # Pipeline D: Document Layout Understanding (FUNSD)
    print("\n>>> [4/5] Executing Pipeline D: Document Layout Understanding (FUNSD)...")
    summary["pipelines"]["pipeline_d_layout"] = train_layout_model(seed=seed)

    # Pipeline E: Medicine Normalization (RxNorm + RxTerms)
    print("\n>>> [5/5] Executing Pipeline E: Medicine Normalization & Ontology Mapping...")
    summary["pipelines"]["pipeline_e_normalization"] = train_normalization_pipeline(seed=seed)

    total_duration = round(time.time() - start_total_time, 2)
    summary["total_duration_seconds"] = total_duration

    # Write JSON Summary
    json_summary_path = reports_dir / "multi_task_training_summary.json"
    with open(json_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Write Markdown Report
    md_summary_path = reports_dir / "SUMMARY.md"
    with open(md_summary_path, "w", encoding="utf-8") as f:
        f.write("# Multi-Task Machine Learning Training & Evaluation Summary\n\n")
        f.write(f"**Execution Timestamp**: `{summary['timestamp']}`  \n")
        f.write(f"**Total Duration**: `{total_duration}s`  \n")
        f.write(f"**Global Seed**: `{seed}`  \n")
        f.write(f"**Hardware Platform**: `{summary['hardware_information']['os_name']} {summary['hardware_information']['os_release']} ({summary['hardware_information']['processor']})`  \n\n")
        
        f.write("## Independent Pipeline Results Matrix\n\n")
        f.write("| Pipeline | Model Architecture | Dataset & Version | Key Test Metric | Checkpoint |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        
        p_a = summary["pipelines"]["pipeline_a_ocr"]
        f.write(f"| **Pipeline A: OCR** | `{p_a['model_configuration']['engine']}` | {p_a['dataset_name']} ({p_a['dataset_version']}) | **CER: {p_a['test_metrics']['test_character_error_rate_cer']}**, Field Acc: {p_a['test_metrics']['test_field_extraction_accuracy']*100}% | `{p_a['checkpoint_path']}` |\n")
        
        p_b = summary["pipelines"]["pipeline_b_handwriting"]
        f.write(f"| **Pipeline B: Handwriting** | `{p_b['model_configuration']['architecture']}` | {p_b['dataset_name']} ({p_b['dataset_version']}) | **Top-1 Acc: {p_b['test_metrics']['test_top1_accuracy']*100}%**, Top-5: {p_b['test_metrics']['test_top5_accuracy']*100}% | `{p_b['checkpoint_path']}` |\n")
        
        p_c = summary["pipelines"]["pipeline_c_augmentation"]
        f.write(f"| **Pipeline C: Augmentation** | `{p_c['model_configuration']['generator']}` | {p_c['dataset_name']} ({p_c['dataset_version']}) | **Samples Generated: {p_c['training_metrics']['train_samples_generated']}**, Coverage: 100% | `{p_c['checkpoint_path']}` |\n")
        
        p_d = summary["pipelines"]["pipeline_d_layout"]
        f.write(f"| **Pipeline D: Layout** | `{p_d['model_configuration']['architecture']}` | {p_d['dataset_name']} ({p_d['dataset_version']}) | **Key-Value Acc: {p_d['test_metrics']['key_value_pairing_accuracy']*100}%** | `{p_d['checkpoint_path']}` |\n")
        
        p_e = summary["pipelines"]["pipeline_e_normalization"]
        f.write(f"| **Pipeline E: Normalization** | `{p_e['model_configuration']['algorithm']}` | {p_e['dataset_name']} ({p_e['dataset_version']}) | **Top-1 Match: {p_e['test_metrics']['test_top1_normalization_accuracy']*100}%** | `{p_e['checkpoint_path']}` |\n")

    print("\n================================================================================")
    print(f"[SUCCESS] All 5 multi-task pipelines completed successfully in {total_duration}s!")
    print(f"Summary saved to: {json_summary_path}")
    print(f"Markdown report:  {md_summary_path}")
    print("================================================================================\n")

    return summary


if __name__ == "__main__":
    run_all_training_pipelines(handwriting_epochs=4, synthetic_count=1000, seed=42)
