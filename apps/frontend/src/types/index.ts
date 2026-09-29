/**
 * TypeScript Definitions for AI Medicine Assistant Frontend (Aligned with Backend Pydantic Schemas)
 */

export type VerificationStatus = "verified" | "review_required" | "unverified";
export type SafetyLevel = "LOW" | "MODERATE" | "HIGH" | "EMERGENCY" | "OUT_OF_SCOPE";
export type InteractionSeverity = "CONTRAINDICATED" | "MAJOR" | "MODERATE" | "MINOR" | "UNKNOWN";

export type IntentType =
  | "MEDICINE_INFORMATION"
  | "MEDICINE_USE"
  | "SIDE_EFFECT"
  | "DOSAGE_INFORMATION"
  | "CONTRAINDICATION"
  | "DRUG_INTERACTION"
  | "MISSED_DOSE"
  | "PREGNANCY"
  | "CHILD_MEDICATION"
  | "ELDERLY_MEDICATION"
  | "PRESCRIPTION"
  | "MEDICINE_IDENTIFICATION"
  | "SYMPTOM"
  | "EMERGENCY"
  | "GENERAL_HEALTH"
  | "OUT_OF_SCOPE";

export type EntityLabel =
  | "DRUG"
  | "BRAND"
  | "ACTIVE_INGREDIENT"
  | "DISEASE"
  | "SYMPTOM"
  | "DOSAGE"
  | "FREQUENCY"
  | "DURATION"
  | "AGE"
  | "PREGNANCY"
  | "ALLERGY"
  | "LAB_TEST"
  | "MEDICAL_PROCEDURE";

export interface PrescriptionCandidate {
  raw_text: string;
  normalized_name: string | null;
  rxcui: string | null;
  ingredient: string | null;
  strength: string | null;
  dose: string | null;
  frequency: string | null;
  duration: string | null;
  route: string | null;
  confidence: number;
  verification_status: VerificationStatus;
  match_tier: string;
}

export interface PrescriptionMetadata {
  processing_time_ms: number;
  models_used: string[];
  ocr_method: string;
  normalization_source: string;
  warning_count: number;
  requires_human_review: boolean;
}

export interface PrescriptionResult {
  prescription_id: string;
  patient_info: {
    name?: string | null;
    age?: string | null;
    gender?: string | null;
  };
  doctor_info: {
    name?: string | null;
    registration_number?: string | null;
    clinic?: string | null;
  };
  date: string | null;
  candidates: PrescriptionCandidate[];
  instructions: string[];
  warnings: string[];
  metadata: PrescriptionMetadata;
}

export interface MedicalEntity {
  text: string;
  label: EntityLabel;
  start_char: number;
  end_char: number;
  normalized_rxcui?: string | null;
  canonical_name?: string | null;
  confidence: number;
}

export interface RetrievalResult {
  document_id: string;
  chunk_id?: string | null;
  title: string;
  source: string;
  source_url?: string | null;
  section_name?: string | null;
  text: string;
  score: number;
  retrieval_method: string;
  metadata: Record<string, any>;
}

export interface Citation {
  citation_id: string;
  document_id: string;
  chunk_id: string;
  source: string;
  title: string;
  section: string;
  excerpt: string;
}

export interface DrugInteractionResult {
  drug_a: string;
  drug_b: string;
  drug_a_rxcui?: string | null;
  drug_b_rxcui?: string | null;
  interaction_found: boolean;
  severity?: InteractionSeverity | null;
  clinical_effect?: string | null;
  mechanism?: string | null;
  evidence_source?: string | null;
  recommendation?: string | null;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string | null;
  prescription_context?: PrescriptionResult | Record<string, any> | null;
}

export interface ChatResponse {
  response: string;
  conversation_id: string;
  intent: IntentType;
  safety_level: SafetyLevel;
  citations: Citation[];
  evidence: RetrievalResult[];
  entities: MedicalEntity[];
  interactions: DrugInteractionResult[];
  requires_professional_review: boolean;
  warnings: string[];
}

export interface SystemHealth {
  status: "healthy" | "unhealthy";
  app_name: string;
  environment: string;
  api_prefix: string;
}

export interface SystemReadiness {
  status: "ready" | "not_ready";
  models_loaded: boolean;
  metadata?: Record<string, any>;
}
