import React, { useState } from "react";
import Head from "next/head";
import { UploadZone } from "../components/Prescription/UploadZone";
import { PrescriptionViewer } from "../components/Prescription/PrescriptionViewer";
import { ChatInterface } from "../components/Chat/ChatInterface";
import { InteractionChecker } from "../components/Interactions/InteractionChecker";
import { PrescriptionResult } from "../types";

export default function ClinicalWorkspacePage() {
  const [activeTab, setActiveTab] = useState<"prescription" | "chat" | "interactions" | "safety">("prescription");
  const [prescription, setPrescription] = useState<PrescriptionResult | null>(null);

  return (
    <>
      <Head>
        <title>AI Medicine Assistant — Clinical Workspace</title>
        <meta name="description" content="Prescription OCR, RxNorm Verification, and Evidence-Grounded Medical RAG" />
      </Head>

      <div className="app-body">
        <header className="top-nav">
          <div className="brand-details">
            <h1 className="brand-title">AI MEDICINE ASSISTANT</h1>
            <span className="brand-subtitle">Clinical OCR • RxNorm Verification • Evidence RAG</span>
          </div>

          <nav className="nav-tabs">
            <button
              className={`nav-tab ${activeTab === "prescription" ? "active" : ""}`}
              onClick={() => setActiveTab("prescription")}
            >
              Prescriptions
            </button>
            <button
              className={`nav-tab ${activeTab === "chat" ? "active" : ""}`}
              onClick={() => setActiveTab("chat")}
            >
              Clinical Chat
            </button>
            <button
              className={`nav-tab ${activeTab === "interactions" ? "active" : ""}`}
              onClick={() => setActiveTab("interactions")}
            >
              Drug Interactions
            </button>
          </nav>
        </header>

        <main className="main-container">
          {activeTab === "prescription" && (
            <div className="workspace-grid">
              <UploadZone onPrescriptionExtracted={(res) => setPrescription(res)} />
              <PrescriptionViewer
                prescription={prescription}
                onAskChat={() => setActiveTab("chat")}
              />
            </div>
          )}

          {activeTab === "chat" && (
            <ChatInterface
              prescriptionContext={prescription}
              onClearContext={() => setPrescription(null)}
              onOpenCitation={(id) => alert(`Citation Source: ${id}`)}
            />
          )}

          {activeTab === "interactions" && <InteractionChecker />}
        </main>
      </div>
    </>
  );
}
