/**
 * Client-Side API Communication Module for AI Medicine Assistant
 */

const API_BASE_URL = '/api/v1';

class MedicineApiClient {
  constructor(baseUrl = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const headers = options.headers || {};
    
    if (!options.isFormData && !headers['Content-Type']) {
      headers['Content-Type'] = 'application/json';
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        const errorMsg = data.detail || data.message || `Request failed with status ${response.status}`;
        const error = new Error(errorMsg);
        error.status = response.status;
        error.details = data;
        throw error;
      }

      return data;
    } catch (err) {
      if (err.name === 'TypeError' && err.message.includes('fetch')) {
        const netErr = new Error('Cannot connect to backend server. Please ensure FastAPI service is running on port 8000.');
        netErr.status = 0;
        throw netErr;
      }
      throw err;
    }
  }

  async checkHealth() {
    return this.request('/health', { method: 'GET' });
  }

  async checkReadiness() {
    return this.request('/ready', { method: 'GET' });
  }

  async uploadAndExtractPrescription(file) {
    const formData = new FormData();
    formData.append('file', file);

    return this.request('/prescriptions/upload-and-extract', {
      method: 'POST',
      body: formData,
      isFormData: true
    });
  }

  async sendChatMessage(message, conversationId = null, prescriptionContext = null) {
    return this.request('/chat', {
      method: 'POST',
      body: JSON.stringify({
        message,
        conversation_id: conversationId,
        prescription_context: prescriptionContext
      })
    });
  }
}

// Global API Client Instance
window.apiClient = new MedicineApiClient();
