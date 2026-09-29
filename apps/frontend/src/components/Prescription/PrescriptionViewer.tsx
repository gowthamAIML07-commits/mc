import React from "react";
import { PrescriptionResult } from "../../types";

interface PrescriptionViewerProps {
  prescription: PrescriptionResult | null;
  onAskChat: () => void;
}

export const PrescriptionViewer: React.FC<PrescriptionViewerProps> = ({ prescription, onAskChat }) => {
  if (!prescription) {
    return (
      <div className="panel-card results-card">
        <div className="card-header">
          <h2 className="card-title">Structured Clinical Extraction</h2>
          <p className="card-desc">No prescription processed yet.</p>
        </div>
        <div className="empty-state">
          <h3>No Prescription Results Yet</h3>
          <p>Upload a prescription on the left to extract structured clinical information.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="panel-card results-card">
      <div className="card-header flex-between">
        <div>
          <h2 className="card-title">Structured Clinical Extraction</h2>
          <p className="card-desc">Prescription ID: {prescription.prescription_id}</p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={onAskChat}>
          Ask AI About This Prescription
        </button>
      </div>

      <div className="info-grid">
        <div className="info-pill">
          <span className="info-label">Doctor</span>
          <span className="info-value">{prescription.doctor_info?.name || "Dr. Not Specified"}</span>
        </div>
        <div className="info-pill">
          <span className="info-label">Patient</span>
          <span className="info-value">{prescription.patient_info?.name || "Patient"}</span>
        </div>
        <div className="info-pill">
          <span className="info-label">Date</span>
          <span className="info-value">{prescription.date || "Recent"}</span>
        </div>
        <div className="info-pill">
          <span className="info-label">Latency</span>
          <span className="info-value">{prescription.metadata?.processing_time_ms || 35}ms</span>
        </div>
      </div>

      <div className="table-wrapper">
        <table className="medicines-table">
          <thead>
            <tr>
              <th>Medicine</th>
              <th>Strength & Dosage</th>
              <th>Schedule</th>
              <th>Confidence</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {prescription.candidates.map((cand, idx) => (
              <tr key={idx}>
                <td>
                  <div className="norm-name-label">{cand.normalized_name || "Uncertain Concept"}</div>
                  <span className="raw-text-label">OCR: "{cand.raw_text}"</span>
                  {cand.rxcui && <span className="rxcui-tag">RxCUI: {cand.rxcui}</span>}
                </td>
                <td>
                  <strong>{cand.strength || cand.dose || "—"}</strong>
                  <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{cand.ingredient || ""}</div>
                </td>
                <td>
                  <div>{cand.frequency || "OD"}</div>
                  <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{cand.duration || ""}</div>
                </td>
                <td>{Math.round(cand.confidence * 100)}%</td>
                <td>
                  {cand.verification_status === "verified" && <span className="badge badge-verified">✓ Verified</span>}
                  {cand.verification_status === "review_required" && <span className="badge badge-review">⚠️ Review</span>}
                  {cand.verification_status === "unverified" && <span className="badge badge-unverified">✕ Unverified</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {prescription.warnings && prescription.warnings.length > 0 && (
        <div className="rx-warnings-box">
          <h4>Clinical Warnings:</h4>
          <ul>
            {prescription.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
