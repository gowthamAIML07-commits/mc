import React, { useState, useRef } from "react";
import { apiClient } from "../../api/client";
import { PrescriptionResult } from "../../types";

interface UploadZoneProps {
  onPrescriptionExtracted: (result: PrescriptionResult) => void;
}

export const UploadZone: React.FC<UploadZoneProps> = ({ onPrescriptionExtracted }) => {
  const [dragOver, setDragOver] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = (file: File) => {
    const validTypes = ["image/png", "image/jpeg", "image/jpg", "image/webp"];
    if (!validTypes.includes(file.type)) {
      setError("Unsupported format. Please select PNG, JPEG, or WEBP.");
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      setError("File size exceeds 15MB limit.");
      return;
    }
    setError(null);
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleProcess = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.uploadPrescription(selectedFile);
      onPrescriptionExtracted(res);
    } catch (err: any) {
      setError(err.message || "Prescription extraction failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="panel-card upload-card">
      <div className="card-header">
        <h2 className="card-title">Prescription Upload & Recognition</h2>
        <p className="card-desc">CRNN Vision OCR • Spatial Layout • RxNorm Verification</p>
      </div>

      <div className="privacy-banner">
        <strong>Privacy Guaranteed:</strong> Prescription images are processed in-memory and are not permanently stored.
      </div>

      <div
        className={`dropzone ${dragOver ? "dragover" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
        }}
        onClick={() => !previewUrl && fileInputRef.current?.click()}
      >
        <input
          type="file"
          ref={fileInputRef}
          accept="image/png,image/jpeg,image/jpg,image/webp"
          style={{ display: "none" }}
          onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        />

        {!previewUrl ? (
          <div className="dropzone-idle">
            <p className="dropzone-main-text">Drag & drop prescription image here, or browse file</p>
            <p className="dropzone-sub-text">PNG, JPEG, WEBP up to 15MB</p>
          </div>
        ) : (
          <div className="dropzone-preview">
            <img src={previewUrl} alt="Prescription preview" />
            <div className="preview-actions">
              <button className="btn btn-secondary btn-sm" onClick={(e) => { e.stopPropagation(); handleReset(); }}>
                Remove
              </button>
              <button
                className="btn btn-primary btn-sm"
                onClick={(e) => { e.stopPropagation(); handleProcess(); }}
                disabled={loading}
              >
                {loading ? "Processing..." : "Process Prescription"}
              </button>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="alert-banner alert-error" style={{ marginTop: "1rem" }}>
          <span>{error}</span>
        </div>
      )}
    </div>
  );
};
