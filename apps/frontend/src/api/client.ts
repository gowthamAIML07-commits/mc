/**
 * Typed API Client for AI Medicine Assistant Backend
 */
import axios, { AxiosInstance, AxiosError } from "axios";
import {
  ChatRequest,
  ChatResponse,
  PrescriptionResult,
  SystemHealth,
  SystemReadiness
} from "../types";

export class ApiError extends Error {
  public statusCode?: number;
  public details?: any;

  constructor(message: string, statusCode?: number, details?: any) {
    super(message);
    this.name = "ApiError";
    this.statusCode = statusCode;
    this.details = details;
  }
}

export class MedicineApiClient {
  private client: AxiosInstance;

  constructor(baseURL: string = "/api/v1") {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        "Accept": "application/json"
      }
    });

    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        const status = error.response?.status;
        const data: any = error.response?.data;
        const msg = data?.detail || data?.message || error.message || "An unexpected error occurred.";
        return Promise.reject(new ApiError(msg, status, data));
      }
    );
  }

  /**
   * Upload prescription image and execute multi-task OCR + Normalization extraction.
   */
  async uploadPrescription(file: File): Promise<PrescriptionResult> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await this.client.post<PrescriptionResult>(
      "/prescriptions/upload-and-extract",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data"
        }
      }
    );
    return response.data;
  }

  /**
   * Send a query to the Clinical RAG Chatbot engine.
   */
  async sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
    const response = await this.client.post<ChatResponse>("/chat", request);
    return response.data;
  }

  /**
   * System Liveness Health Check.
   */
  async getHealth(): Promise<SystemHealth> {
    const response = await this.client.get<SystemHealth>("/health");
    return response.data;
  }

  /**
   * System Readiness Probe checking model loading.
   */
  async getReadiness(): Promise<SystemReadiness> {
    const response = await this.client.get<SystemReadiness>("/ready");
    return response.data;
  }
}

export const apiClient = new MedicineApiClient();
