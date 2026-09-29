import React, { useState } from "react";
import { apiClient } from "../../api/client";
import { ChatResponse } from "../../types";

export const InteractionChecker: React.FC = () => {
  const [drugA, setDrugA] = useState("");
  const [drugB, setDrugB] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ChatResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleCheck = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!drugA.trim() || !drugB.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const resp = await apiClient.sendChatMessage(`Can I take ${drugA} together with ${drugB}?`);
      setResult(resp);
    } catch (err: any) {
      setError(err.message || "Failed to check drug interaction.");
    } finally {
      setLoading(false);
    }
  };

  const interaction = result?.interactions?.find((i) => i.interaction_found);

  return (
    <div className="panel-card checker-card">
      <div className="card-header">
        <h2 className="card-title">Structured Drug Interaction Checker</h2>
        <p className="card-desc">Deterministic lookup backed by DailyMed / RxNorm official monographs.</p>
      </div>

      <form className="checker-form" onSubmit={handleCheck}>
        <div className="checker-inputs-grid">
          <div className="form-group">
            <label className="form-label">Primary Medication (Drug A):</label>
            <input
              type="text"
              className="form-input"
              placeholder="e.g. Warfarin"
              value={drugA}
              onChange={(e) => setDrugA(e.target.value)}
              required
            />
          </div>
          <div className="form-group">
            <label className="form-label">Co-Administered Medication (Drug B):</label>
            <input
              type="text"
              className="form-input"
              placeholder="e.g. Aspirin"
              value={drugB}
              onChange={(e) => setDrugB(e.target.value)}
              required
            />
          </div>
        </div>

        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? "Checking Database..." : "Verify Drug Interaction Safety"}
        </button>
      </form>

      {error && (
        <div className="alert-banner alert-error" style={{ marginTop: "1rem" }}>
          <span>{error}</span>
        </div>
      )}

      {result && (
        <div className="interaction-result-box">
          {interaction ? (
            <div>
              <h3>
                Verified Interaction: {interaction.drug_a} + {interaction.drug_b}{" "}
                <span className="badge badge-unverified">🚨 {interaction.severity}</span>
              </h3>
              <p><strong>Clinical Effect:</strong> {interaction.clinical_effect}</p>
              <p><strong>Mechanism:</strong> {interaction.mechanism}</p>
              <p><strong>Recommendation:</strong> {interaction.recommendation}</p>
            </div>
          ) : (
            <div style={{ color: "var(--success)" }}>
              <h3>✓ No Verified Major Interaction Documented</h3>
              <p>No high-risk interaction between {drugA} and {drugB} was found in the authoritative clinical database.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
