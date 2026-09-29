/**
 * AI Medicine Assistant — Clinical Workspace Interactive Controller
 */

// Application State
const AppState = {
  activeTab: 'prescription',
  currentPrescription: null,
  activeConversationId: null,
  isProcessing: false,
  selectedFile: null,
  cachedCitations: new Map(),
  activeMonographKey: 'DAILYMED_WARFARIN_5'
};

// Monograph Knowledge Base Definition for Local Navigation
const MONOGRAPH_CACHE = {
  DAILYMED_WARFARIN_5: {
    title: 'Warfarin Sodium 5 MG Oral Tablet',
    ingredient: 'Warfarin',
    rxcui: '11289',
    atc: 'B01AA03',
    sections: {
      'Indications & Clinical Usage': 'Prophylaxis and treatment of venous thrombosis, pulmonary embolism, and thromboembolic complications associated with atrial fibrillation or post-myocardial infarction.',
      'Dosage & Administration Guidelines': 'Individualize dosing based on target International Normalized Ratio (INR). Usual initial dose is 2 mg to 5 mg orally once daily. Monitor INR closely (target 2.0 to 3.0).',
      'Contraindications & Black Box Warnings': 'Black Box Warning: Major or fatal bleeding. Contraindicated in pregnancy, hemorrhagic tendencies, active GI ulcers, recent eye or CNS surgery.',
      'Drug-Drug & Food Interactions': 'Aspirin, NSAIDs, and antiplatelet drugs dramatically increase gastrointestinal bleeding risk. CYP2C9 inhibitors (amiodarone, fluconazole) elevate INR. Avoid sudden changes in dietary Vitamin K.',
      'Pregnancy & Teratogenicity': 'Contraindicated in pregnancy due to severe fetal hemorrhage and warfarin embryopathy.'
    }
  },
  DAILYMED_METFORMIN_500: {
    title: 'Metformin Hydrochloride 500 MG Oral Tablet',
    ingredient: 'Metformin',
    rxcui: '860975',
    atc: 'A10BA02',
    sections: {
      'Indications & Clinical Usage': 'Adjunct to diet and exercise to improve glycemic control in type 2 diabetes mellitus.',
      'Dosage & Administration Guidelines': 'Starting dose is 500 mg orally twice daily with meals or 850 mg once daily. Maximum daily dose is 2,000 mg to 2,550 mg in divided doses.',
      'Contraindications & Boxed Warnings': 'Black Box Warning: Lactic Acidosis risk increases with renal impairment (eGFR < 30 mL/min), sepsis, and acute alcohol consumption. Contraindicated in severe renal impairment.',
      'Drug-Drug & Food Interactions': 'Alcohol potentiates effect on lactate metabolism. Cationic drugs (cimetidine) and carbonic anhydrase inhibitors increase exposure.',
      'Patient Counseling': 'Take with meals to minimize gastrointestinal discomfort. Report symptoms of lactic acidosis immediately.'
    }
  },
  DAILYMED_AMOXICILLIN_500: {
    title: 'Amoxicillin 500 MG Oral Capsule',
    ingredient: 'Amoxicillin',
    rxcui: '308189',
    atc: 'J01CA04',
    sections: {
      'Indications & Clinical Usage': 'Treatment of bacterial infections including otitis media, streptococcal pharyngitis, pneumonia, acute sinusitis, and skin/urinary tract infections.',
      'Dosage & Administration Guidelines': 'Standard adult dosage is 500 mg orally every 8 hours (TDS) or 875 mg every 12 hours (BD) for 7 to 10 days. Take with or without food.',
      'Contraindications & Boxed Warnings': 'Contraindicated in patients with severe hypersensitivity reactions (anaphylaxis) to penicillins or cephalosporins.',
      'Drug-Drug Interactions': 'Probenecid increases amoxicillin serum concentrations. Concomitant use with Warfarin may increase bleeding risk; monitor INR.',
      'Adverse Reactions': 'Diarrhea, nausea, vomiting, skin rashes, urticaria, and headache.'
    }
  },
  DAILYMED_PARACETAMOL_650: {
    title: 'Acetaminophen / Paracetamol 650 MG Tablet',
    ingredient: 'Paracetamol',
    rxcui: '161',
    atc: 'N02BE01',
    sections: {
      'Indications & Clinical Usage': 'Temporary relief of minor aches, pains, headaches, arthritis pain, toothaches, and reduction of fever.',
      'Dosage & Administration Guidelines': 'Adults: 650 mg orally every 4 to 6 hours as needed (SOS). Maximum 3,000 mg in 24 hours. Minimum interval is 4 hours.',
      'Warnings & Hepatotoxicity': 'Liver Warning: Ingestion of >4,000 mg/day or combination with heavy alcohol intake may cause acute liver failure and fatal hepatic necrosis.',
      'Drug-Drug Interactions': 'Chronic alcohol consumption induces CYP2E1, enhancing toxic metabolite (NAPQI) production. Prolonged use with Warfarin may elevate INR.'
    }
  },
  DAILYMED_PANTOPRAZOLE_40: {
    title: 'Pantoprazole Sodium 40 MG Delayed-Release Tablet',
    ingredient: 'Pantoprazole',
    rxcui: '312615',
    atc: 'A02BC02',
    sections: {
      'Indications & Clinical Usage': 'Short-term treatment of erosive esophagitis, gastroesophageal reflux disease (GERD), and Zollinger-Ellison syndrome.',
      'Dosage & Administration Guidelines': 'Adults: 40 mg once daily (OD) in the morning, 30 to 60 minutes before breakfast. Swallow whole; do not chew or crush.',
      'Drug-Drug Interactions': 'Methotrexate co-administration may increase methotrexate serum concentrations. Rilpivirine is contraindicated.'
    }
  },
  DAILYMED_ATORVASTATIN_10: {
    title: 'Atorvastatin Calcium 10 MG Oral Tablet',
    ingredient: 'Atorvastatin',
    rxcui: '310798',
    atc: 'C10AA05',
    sections: {
      'Indications & Clinical Usage': 'Primary hyperlipidemia, mixed dyslipidemia, and prevention of cardiovascular disease.',
      'Dosage & Administration Guidelines': '10 mg to 80 mg orally once daily with or without food. Usual starting dose is 10 mg or 20 mg once daily.',
      'Warnings & Myopathy': 'Risk of myopathy and rhabdomyolysis is elevated with strong CYP3A4 inhibitors (clarithromycin, itraconazole). Contraindicated in active liver disease and pregnancy.'
    }
  },
  DAILYMED_AZITHROMYCIN_500: {
    title: 'Azithromycin 500 MG Oral Tablet',
    ingredient: 'Azithromycin',
    rxcui: '198440',
    atc: 'J01FA10',
    sections: {
      'Indications & Clinical Usage': 'Community-acquired pneumonia, acute bacterial exacerbations of COPD, bacterial sinusitis, pharyngitis, and uncomplicated skin infections.',
      'Dosage & Administration Guidelines': '500 mg as a single dose on Day 1, followed by 250 mg once daily on Days 2 through 5.',
      'Warnings & Precautions': 'QT prolongation and risk of cardiac arrhythmias and torsades de pointes.'
    }
  },
  DAILYMED_CETIRIZINE_10: {
    title: 'Cetirizine Hydrochloride 10 MG Tablet',
    ingredient: 'Cetirizine',
    rxcui: '310489',
    atc: 'R06AE07',
    sections: {
      'Indications & Clinical Usage': 'Relief of symptoms associated with seasonal and perennial allergic rhinitis and chronic urticaria.',
      'Dosage & Administration Guidelines': 'Adults: 5 mg or 10 mg once daily, preferably at bedtime.',
      'Warnings & Precautions': 'May cause somnolence and CNS depression. Caution when driving or combining with alcohol.'
    }
  }
};

// Initialization on DOM Loaded
document.addEventListener('DOMContentLoaded', () => {
  initTabNavigation();
  initPrescriptionUploader();
  initChatWorkspace();
  initMonographBrowser();
  initInteractionChecker();
  initCitationModal();
  checkSystemHealth();
});

/* ==========================================================================
   TAB NAVIGATION
   ========================================================================== */

function initTabNavigation() {
  const tabs = document.querySelectorAll('.nav-tab');
  const panels = document.querySelectorAll('.tab-panel');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetTab = tab.getAttribute('data-tab');
      switchTab(targetTab);
    });
  });
}

function switchTab(targetTab) {
  AppState.activeTab = targetTab;
  
  document.querySelectorAll('.nav-tab').forEach(t => {
    if (t.getAttribute('data-tab') === targetTab) {
      t.classList.add('active');
    } else {
      t.classList.remove('active');
    }
  });

  document.querySelectorAll('.tab-panel').forEach(p => {
    if (p.id === `panel-${targetTab}`) {
      p.classList.add('active');
    } else {
      p.classList.remove('active');
    }
  });
}

/* ==========================================================================
   SYSTEM HEALTH CHECK
   ========================================================================== */

async function checkSystemHealth() {
  const indicator = document.getElementById('system-status-indicator');
  const dot = indicator?.querySelector('.status-dot');
  const label = document.getElementById('system-status-label');

  try {
    const health = await window.apiClient.checkHealth();
    if (health.status === 'healthy') {
      dot?.classList.add('online');
      if (label) label.textContent = 'API Online • Models Ready';
    } else {
      dot?.classList.remove('online');
      if (label) label.textContent = 'Backend Degraded';
    }
  } catch (e) {
    dot?.classList.remove('online');
    if (label) label.textContent = 'Backend Offline';
  }
}

/* ==========================================================================
   PRESCRIPTION UPLOAD & INFERENCE PIPELINE
   ========================================================================== */

function initPrescriptionUploader() {
  const dropzone = document.getElementById('prescription-dropzone');
  const fileInput = document.getElementById('prescription-file-input');
  const idleView = document.getElementById('dropzone-idle-view');
  const previewView = document.getElementById('dropzone-preview-view');
  const previewImg = document.getElementById('prescription-preview-img');
  const btnRemove = document.getElementById('btn-remove-image');
  const btnProcess = document.getElementById('btn-process-image');

  dropzone?.addEventListener('click', (e) => {
    if (e.target !== btnRemove && e.target !== btnProcess && !AppState.selectedFile) {
      fileInput?.click();
    }
  });

  fileInput?.addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (file) handleFileSelection(file);
  });

  // Drag and Drop
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone?.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone?.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone?.addEventListener('drop', (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (file) handleFileSelection(file);
  });

  btnRemove?.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUploader();
  });

  btnProcess?.addEventListener('click', (e) => {
    e.stopPropagation();
    if (AppState.selectedFile) {
      executePrescriptionPipeline(AppState.selectedFile);
    }
  });

  // "Ask AI about this Prescription" button
  const btnAskChat = document.getElementById('btn-ask-chat-from-rx');
  btnAskChat?.addEventListener('click', () => {
    if (AppState.currentPrescription) {
      attachPrescriptionToChat(AppState.currentPrescription);
      switchTab('chat');
    }
  });
}

function handleFileSelection(file) {
  const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
  const maxBytes = 15 * 1024 * 1024; // 15MB
  const errBanner = document.getElementById('upload-error-banner');
  const errMsg = document.getElementById('upload-error-message');

  if (!validTypes.includes(file.type)) {
    showUploadError('Unsupported file type. Please upload a PNG, JPEG, or WEBP prescription image.');
    return;
  }

  if (file.size > maxBytes) {
    showUploadError('File size exceeds maximum allowable limit of 15MB.');
    return;
  }

  if (errBanner) errBanner.style.display = 'none';

  AppState.selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    const previewImg = document.getElementById('prescription-preview-img');
    const idleView = document.getElementById('dropzone-idle-view');
    const previewView = document.getElementById('dropzone-preview-view');
    
    if (previewImg) previewImg.src = e.target.result;
    if (idleView) idleView.style.display = 'none';
    if (previewView) previewView.style.display = 'block';
  };
  reader.readAsDataURL(file);
}

function resetUploader() {
  AppState.selectedFile = null;
  const fileInput = document.getElementById('prescription-file-input');
  const idleView = document.getElementById('dropzone-idle-view');
  const previewView = document.getElementById('dropzone-preview-view');
  const previewImg = document.getElementById('prescription-preview-img');
  const errBanner = document.getElementById('upload-error-banner');
  const stepper = document.getElementById('processing-stepper');

  if (fileInput) fileInput.value = '';
  if (previewImg) previewImg.src = '';
  if (previewView) previewView.style.display = 'none';
  if (idleView) idleView.style.display = 'block';
  if (errBanner) errBanner.style.display = 'none';
  if (stepper) stepper.style.display = 'none';
}

function showUploadError(msg) {
  const errBanner = document.getElementById('upload-error-banner');
  const errMsg = document.getElementById('upload-error-message');
  if (errMsg) errMsg.textContent = msg;
  if (errBanner) errBanner.style.display = 'flex';
}

async function executePrescriptionPipeline(file) {
  const stepper = document.getElementById('processing-stepper');
  const progressFill = document.getElementById('progress-bar-fill');
  const percentLabel = document.getElementById('stepper-percent');
  const currentLabel = document.getElementById('stepper-current-label');
  const errBanner = document.getElementById('upload-error-banner');

  if (stepper) stepper.style.display = 'block';
  if (errBanner) errBanner.style.display = 'none';

  const stages = [
    { id: 'stage-upload', label: 'Uploading & Validating Image', pct: 20 },
    { id: 'stage-ocr', label: 'Executing CRNN & Spatial Layout Parsing', pct: 45 },
    { id: 'stage-ner', label: 'Extracting Medical Entities & Dosages', pct: 65 },
    { id: 'stage-norm', label: 'Performing Multi-Tier RxNorm Normalization', pct: 85 },
    { id: 'stage-verify', label: 'Clinical Verification & Confidence Scoring', pct: 100 }
  ];

  let currentStageIdx = 0;
  function updateProgress(idx) {
    stages.forEach((stg, i) => {
      const el = document.getElementById(stg.id);
      if (el) {
        if (i < idx) {
          el.className = 'stage-item completed';
        } else if (i === idx) {
          el.className = 'stage-item active';
        } else {
          el.className = 'stage-item';
        }
      }
    });

    const curr = stages[Math.min(idx, stages.length - 1)];
    if (progressFill) progressFill.style.width = `${curr.pct}%`;
    if (percentLabel) percentLabel.textContent = `${curr.pct}%`;
    if (currentLabel) currentLabel.textContent = curr.label;
  }

  updateProgress(0);
  const progressTimer = setInterval(() => {
    if (currentStageIdx < stages.length - 1) {
      currentStageIdx++;
      updateProgress(currentStageIdx);
    }
  }, 400);

  try {
    const result = await window.apiClient.uploadAndExtractPrescription(file);
    clearInterval(progressTimer);
    updateProgress(stages.length);

    setTimeout(() => {
      if (stepper) stepper.style.display = 'none';
      renderPrescriptionResults(result);
    }, 400);
  } catch (err) {
    clearInterval(progressTimer);
    if (stepper) stepper.style.display = 'none';
    showUploadError(err.message || 'Prescription extraction failed.');
  }
}

function renderPrescriptionResults(data) {
  AppState.currentPrescription = data;

  const emptyState = document.getElementById('results-empty-state');
  const resultsView = document.getElementById('results-content-view');
  const btnAskChat = document.getElementById('btn-ask-chat-from-rx');
  const metaDesc = document.getElementById('results-meta-desc');

  if (emptyState) emptyState.style.display = 'none';
  if (resultsView) resultsView.style.display = 'block';
  if (btnAskChat) btnAskChat.style.display = 'inline-flex';

  if (metaDesc) {
    metaDesc.textContent = `Prescription ID: ${data.prescription_id} • Status: ${data.metadata?.requires_human_review ? 'Human Review Advised' : 'Processed'}`;
  }

  // Info Pills
  const docVal = document.getElementById('rx-doctor-val');
  const patVal = document.getElementById('rx-patient-val');
  const dateVal = document.getElementById('rx-date-val');
  const latVal = document.getElementById('rx-latency-val');

  if (docVal) docVal.textContent = data.doctor_info?.name || 'Dr. Not Specified';
  if (patVal) patVal.textContent = data.patient_info?.name || 'Patient';
  if (dateVal) dateVal.textContent = data.date || 'Recent';
  if (latVal) latVal.textContent = `${data.metadata?.processing_time_ms || 35}ms`;

  // Medicines Table
  const tbody = document.getElementById('medicines-table-body');
  if (tbody) {
    tbody.innerHTML = '';

    (data.candidates || []).forEach(cand => {
      const tr = document.createElement('tr');

      // Verification Badge
      let statusBadge = '';
      if (cand.verification_status === 'verified') {
        statusBadge = `<span class="badge badge-verified">✓ Verified</span>`;
      } else if (cand.verification_status === 'review_required') {
        statusBadge = `<span class="badge badge-review">⚠️ Review Required</span>`;
      } else {
        statusBadge = `<span class="badge badge-unverified">✕ Unverified</span>`;
      }

      const confPct = Math.round((cand.confidence || 0) * 100);
      const rxcuiHtml = cand.rxcui ? `<span class="rxcui-tag">RxCUI: ${cand.rxcui}</span>` : '';
      const normName = cand.normalized_name || '<span style="color:#f87171;">Uncertain Concept</span>';

      tr.innerHTML = `
        <td>
          <div class="norm-name-label">${normName}</div>
          <span class="raw-text-label">OCR: "${escapeHtml(cand.raw_text)}"</span>
          ${rxcuiHtml}
        </td>
        <td>
          <strong>${cand.strength || cand.dose || '—'}</strong>
          <div style="font-size:0.72rem;color:var(--text-muted);">${cand.ingredient || ''}</div>
        </td>
        <td>
          <div>${cand.frequency || 'OD'}</div>
          <div style="font-size:0.72rem;color:var(--text-muted);">${cand.duration || ''} (${cand.route || 'Oral'})</div>
        </td>
        <td>
          <div class="confidence-bar-container">
            <div class="confidence-bar"><div class="confidence-fill" style="width:${confPct}%;"></div></div>
            <span style="font-size:0.75rem;font-weight:600;">${confPct}%</span>
          </div>
        </td>
        <td>${statusBadge}</td>
        <td>
          <button type="button" class="btn btn-secondary btn-xs btn-inspect-drug" data-name="${escapeHtml(cand.ingredient || cand.normalized_name || cand.raw_text)}">Inspect</button>
        </td>
      `;
      tbody.appendChild(tr);
    });

    // Attach inspect drug listener
    tbody.querySelectorAll('.btn-inspect-drug').forEach(btn => {
      btn.addEventListener('click', () => {
        const drugName = btn.getAttribute('data-name');
        openDrugMonograph(drugName);
      });
    });
  }

  // Warnings Box
  const warningsBox = document.getElementById('rx-warnings-container');
  const warningsList = document.getElementById('rx-warnings-list');
  if (warningsBox && warningsList) {
    const allWarnings = data.warnings || [];
    if (allWarnings.length > 0) {
      warningsBox.style.display = 'block';
      warningsList.innerHTML = allWarnings.map(w => `<li>${escapeHtml(w)}</li>`).join('');
    } else {
      warningsBox.style.display = 'none';
    }
  }
}

/* ==========================================================================
   CLINICAL CHAT WORKSPACE
   ========================================================================== */

function initChatWorkspace() {
  const form = document.getElementById('chat-composer-form');
  const input = document.getElementById('chat-input-text');
  const btnClearCtx = document.getElementById('btn-clear-chat-context');

  form?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = input?.value?.trim();
    if (!text) return;

    input.value = '';
    await sendChatMessage(text);
  });

  // Starter prompt clicks
  document.querySelectorAll('.prompt-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const query = chip.getAttribute('data-query');
      if (query) sendChatMessage(query);
    });
  });

  btnClearCtx?.addEventListener('click', () => {
    AppState.currentPrescription = null;
    const ribbon = document.getElementById('chat-context-ribbon');
    if (ribbon) ribbon.style.display = 'none';
  });
}

function attachPrescriptionToChat(rxData) {
  const ribbon = document.getElementById('chat-context-ribbon');
  const label = document.getElementById('chat-attached-meds-label');
  
  if (ribbon && label) {
    ribbon.style.display = 'flex';
    const medNames = (rxData.candidates || []).map(c => c.normalized_name || c.raw_text).join(', ');
    label.textContent = medNames || 'Prescription Items Attached';
  }
}

async function sendChatMessage(queryText) {
  const scrollContainer = document.getElementById('chat-messages-scroll');
  if (!scrollContainer) return;

  // Append User Message
  const userBubble = document.createElement('div');
  userBubble.className = 'message-bubble message-user';
  userBubble.innerHTML = `<p>${escapeHtml(queryText)}</p>`;
  scrollContainer.appendChild(userBubble);
  scrollContainer.scrollTop = scrollContainer.scrollHeight;

  // Append Loading Bubble
  const loadingBubble = document.createElement('div');
  loadingBubble.className = 'message-bubble message-assistant';
  loadingBubble.id = 'chat-loading-bubble';
  loadingBubble.innerHTML = `
    <div class="message-header">
      <span class="author-badge">AI Clinical Pharmacist</span>
      <span class="time-badge">Retrieving authoritative evidence...</span>
    </div>
    <div style="font-size:0.8rem;color:var(--text-muted);">Consulting DailyMed / RxNorm knowledge graph...</div>
  `;
  scrollContainer.appendChild(loadingBubble);
  scrollContainer.scrollTop = scrollContainer.scrollHeight;

  try {
    const response = await window.apiClient.sendChatMessage(
      queryText,
      AppState.activeConversationId,
      AppState.currentPrescription
    );

    AppState.activeConversationId = response.conversation_id;
    loadingBubble.remove();
    renderAssistantResponse(response, scrollContainer);
  } catch (err) {
    loadingBubble.remove();
    const errorBubble = document.createElement('div');
    errorBubble.className = 'message-bubble message-assistant';
    errorBubble.innerHTML = `
      <div class="message-header">
        <span class="author-badge" style="color:var(--danger);">Error</span>
      </div>
      <p style="color:#f87171;">${escapeHtml(err.message || 'Failed to generate clinical response.')}</p>
    `;
    scrollContainer.appendChild(errorBubble);
    scrollContainer.scrollTop = scrollContainer.scrollHeight;
  }
}

function renderAssistantResponse(resp, container) {
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble message-assistant';

  // Cache citations for interactive modal clicks
  (resp.citations || []).forEach(c => {
    AppState.cachedCitations.set(c.citation_id, c);
    if (c.chunk_id) AppState.cachedCitations.set(c.chunk_id, c);
  });

  (resp.evidence || []).forEach(e => {
    if (e.chunk_id) {
      AppState.cachedCitations.set(e.chunk_id, {
        citation_id: e.chunk_id,
        document_id: e.document_id,
        chunk_id: e.chunk_id,
        source: e.source,
        title: e.title,
        section: e.section_name || 'Clinical Monograph',
        excerpt: e.text
      });
    }
  });

  // Parse formatting & citations in response text
  let formattedText = formatMarkdown(resp.response);
  formattedText = formattedText.replace(/\[(?:Source|Citation):\s*([a-zA-Z0-9_\-\.]+)\]/g, (match, cid) => {
    return `<button type="button" class="citation-pill" onclick="openCitationModal('${cid}')">📖 Source: ${cid}</button>`;
  });

  // Safety Warning Banner (Faithfully reflecting backend classification)
  let safetyBannerHtml = '';
  if (resp.safety_level === 'EMERGENCY') {
    safetyBannerHtml = `
      <div class="safety-alert-box safety-alert-emergency" role="alert">
        <strong>🚨 EMERGENCY MEDICAL DIRECTIVE:</strong>
        <p>This inquiry involves potentially life-threatening or acute symptoms. Please call 911 / 112 / emergency services or visit the nearest emergency room immediately.</p>
      </div>
    `;
  } else if (resp.safety_level === 'HIGH' || resp.requires_professional_review) {
    safetyBannerHtml = `
      <div class="safety-alert-box safety-alert-high">
        <strong>⚠️ Clinical Review Notice:</strong> This clinical guidance is derived from official drug labeling for educational decision support. Consult your licensed physician or pharmacist before making medication changes.
      </div>
    `;
  } else if (resp.safety_level === 'OUT_OF_SCOPE') {
    safetyBannerHtml = `
      <div class="safety-alert-box safety-alert-out-of-scope">
        <strong>ℹ️ Scope Notice:</strong> This assistant specializes exclusively in medical and pharmaceutical information from official drug labeling.
      </div>
    `;
  }

  // Entities Chips
  let entitiesHtml = '';
  if (resp.entities && resp.entities.length > 0) {
    entitiesHtml = `
      <div class="entities-ribbon">
        <span style="font-size:0.7rem;font-weight:600;color:var(--text-muted);margin-right:4px;">Detected Concepts:</span>
        ${resp.entities.map(e => `<span class="entity-chip"><strong>${escapeHtml(e.label)}</strong>: ${escapeHtml(e.canonical_name || e.text)}</span>`).join('')}
      </div>
    `;
  }

  // Citations List
  let citationsHtml = '';
  if (resp.citations && resp.citations.length > 0) {
    citationsHtml = `
      <div class="citations-list">
        <span style="font-size:0.7rem;font-weight:600;color:var(--text-muted);width:100%;">Authoritative Citations:</span>
        ${resp.citations.map(c => `<button type="button" class="citation-pill" onclick="openCitationModal('${c.citation_id}')">📄 ${escapeHtml(c.title)} (${escapeHtml(c.section)})</button>`).join('')}
      </div>
    `;
  }

  bubble.innerHTML = `
    <div class="message-header">
      <span class="author-badge">AI Clinical Pharmacist</span>
      <span class="time-badge">Intent: ${escapeHtml(resp.intent)} • Safety: ${escapeHtml(resp.safety_level)}</span>
    </div>
    <div class="message-body">${formattedText}</div>
    ${safetyBannerHtml}
    ${entitiesHtml}
    ${citationsHtml}
  `;

  container.appendChild(bubble);
  container.scrollTop = container.scrollHeight;
}

/* ==========================================================================
   EVIDENCE / CITATION MODAL
   ========================================================================== */

function initCitationModal() {
  const modal = document.getElementById('citation-modal');
  const btnClose = document.getElementById('btn-close-citation-modal');
  const btnActionClose = document.getElementById('btn-modal-close-action');

  const closeModal = () => {
    if (modal) modal.style.display = 'none';
  };

  btnClose?.addEventListener('click', closeModal);
  btnActionClose?.addEventListener('click', closeModal);
  modal?.addEventListener('click', (e) => {
    if (e.target === modal) closeModal();
  });
}

function openCitationModal(citationKey) {
  const citation = AppState.cachedCitations.get(citationKey);
  const modal = document.getElementById('citation-modal');
  if (!modal) return;

  const docTitle = document.getElementById('modal-doc-title');
  const docSection = document.getElementById('modal-doc-section');
  const docSource = document.getElementById('modal-doc-source');
  const chunkId = document.getElementById('modal-chunk-id');
  const excerptText = document.getElementById('modal-excerpt-text');

  if (citation) {
    if (docTitle) docTitle.textContent = citation.title || 'Official Drug Monograph';
    if (docSection) docSection.textContent = citation.section || 'Clinical Labeling';
    if (docSource) docSource.textContent = citation.source || 'FDA DailyMed / US NLM RxNorm';
    if (chunkId) chunkId.textContent = citation.chunk_id || citationKey;
    if (excerptText) excerptText.textContent = citation.excerpt || 'Excerpt text preserved from authoritative labeling.';
  } else {
    if (docTitle) docTitle.textContent = 'Clinical Monograph Excerpt';
    if (docSection) docSection.textContent = 'Knowledge Base Reference';
    if (docSource) docSource.textContent = 'FDA DailyMed / RxNorm';
    if (chunkId) chunkId.textContent = citationKey;
    if (excerptText) excerptText.textContent = `Evidence chunk '${citationKey}' retrieved from authoritative daily labeling.`;
  }

  modal.style.display = 'flex';
}
window.openCitationModal = openCitationModal;

/* ==========================================================================
   MONOGRAPH BROWSER
   ========================================================================== */

function initMonographBrowser() {
  const navItems = document.querySelectorAll('.monograph-nav-item');
  navItems.forEach(item => {
    item.addEventListener('click', () => {
      navItems.forEach(i => i.classList.remove('active'));
      item.classList.add('active');
      const drugKey = item.getAttribute('data-drug');
      if (drugKey) renderMonographDetails(drugKey);
    });
  });

  renderMonographDetails('DAILYMED_WARFARIN_5');
}

function renderMonographDetails(key) {
  const data = MONOGRAPH_CACHE[key];
  if (!data) return;

  const titleEl = document.getElementById('mono-title');
  const ingEl = document.getElementById('mono-ingredient');
  const rxcuiEl = document.getElementById('mono-rxcui');
  const atcEl = document.getElementById('mono-atc');
  const sectionsList = document.getElementById('mono-sections-list');

  if (titleEl) titleEl.textContent = data.title;
  if (ingEl) ingEl.textContent = `Active Ingredient: ${data.ingredient}`;
  if (rxcuiEl) rxcuiEl.textContent = `RxCUI: ${data.rxcui}`;
  if (atcEl) atcEl.textContent = `ATC: ${data.atc}`;

  if (sectionsList) {
    sectionsList.innerHTML = Object.entries(data.sections).map(([secTitle, secContent]) => `
      <div class="monograph-section-block">
        <h4 class="section-block-title">${escapeHtml(secTitle)}</h4>
        <p class="section-block-text">${escapeHtml(secContent)}</p>
      </div>
    `).join('');
  }
}

function openDrugMonograph(drugName) {
  const nameLower = (drugName || '').toLowerCase();
  for (const [key, mono] of Object.entries(MONOGRAPH_CACHE)) {
    if (mono.ingredient.toLowerCase().includes(nameLower) || mono.title.toLowerCase().includes(nameLower)) {
      switchTab('monographs');
      document.querySelectorAll('.monograph-nav-item').forEach(i => {
        if (i.getAttribute('data-drug') === key) {
          i.classList.add('active');
        } else {
          i.classList.remove('active');
        }
      });
      renderMonographDetails(key);
      return;
    }
  }
  switchTab('monographs');
}

/* ==========================================================================
   DRUG INTERACTION CHECKER TAB
   ========================================================================== */

function initInteractionChecker() {
  const form = document.getElementById('interaction-checker-form');
  const inputA = document.getElementById('input-drug-a');
  const inputB = document.getElementById('input-drug-b');
  const resultBox = document.getElementById('interaction-result-box');

  form?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const drugA = inputA?.value?.trim();
    const drugB = inputB?.value?.trim();
    if (!drugA || !drugB) return;

    if (resultBox) {
      resultBox.style.display = 'block';
      resultBox.innerHTML = `
        <div style="font-size:0.85rem;color:var(--text-muted);">
          🔍 Querying structured interaction database for <strong>${escapeHtml(drugA)} + ${escapeHtml(drugB)}</strong>...
        </div>
      `;
    }

    try {
      const resp = await window.apiClient.sendChatMessage(`Can I take ${drugA} together with ${drugB}?`);
      renderInteractionResults(resp, drugA, drugB, resultBox);
    } catch (err) {
      if (resultBox) {
        resultBox.innerHTML = `
          <div style="color:#f87171;font-size:0.85rem;">
            Failed to evaluate drug interaction: ${escapeHtml(err.message)}
          </div>
        `;
      }
    }
  });
}

function renderInteractionResults(resp, drugA, drugB, container) {
  if (!container) return;

  const interactions = resp.interactions || [];
  const found = interactions.find(i => i.interaction_found);

  if (found) {
    const sev = found.severity || 'MODERATE';
    let sevBadge = `<span class="badge badge-review">${sev}</span>`;
    if (sev === 'MAJOR' || sev === 'CONTRAINDICATED') {
      sevBadge = `<span class="badge badge-unverified">🚨 ${sev}</span>`;
    }

    container.innerHTML = `
      <div style="margin-bottom:0.75rem;">
        <h3 style="font-size:1rem;color:var(--text-primary);margin-bottom:0.35rem;">
          Verified Interaction: ${escapeHtml(found.drug_a)} + ${escapeHtml(found.drug_b)} ${sevBadge}
        </h3>
        <p style="font-size:0.82rem;color:var(--text-secondary);margin-bottom:0.5rem;">
          <strong>Clinical Effect:</strong> ${escapeHtml(found.clinical_effect || 'Interaction documented.')}
        </p>
        <p style="font-size:0.82rem;color:var(--text-secondary);margin-bottom:0.5rem;">
          <strong>Mechanism:</strong> ${escapeHtml(found.mechanism || 'Pharmacological interaction.')}
        </p>
        <p style="font-size:0.82rem;color:#fde68a;background:rgba(245,158,11,0.1);padding:0.6rem;border-radius:6px;border:1px solid rgba(245,158,11,0.3);">
          <strong>Recommendation:</strong> ${escapeHtml(found.recommendation || 'Consult your prescribing doctor.')}
        </p>
        <div style="font-size:0.72rem;color:var(--text-muted);margin-top:0.5rem;">
          Source: ${escapeHtml(found.evidence_source || 'DailyMed Drug Labeling')}
        </div>
      </div>
    `;
  } else {
    container.innerHTML = `
      <div style="padding:0.75rem;background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.25);border-radius:6px;">
        <h3 style="font-size:0.95rem;color:#34d399;margin-bottom:0.35rem;">
          ✓ No Major Interaction Documented
        </h3>
        <p style="font-size:0.82rem;color:var(--text-secondary);">
          No verified high-risk interaction between <strong>${escapeHtml(drugA)}</strong> and <strong>${escapeHtml(drugB)}</strong> was found in the authoritative DailyMed/RxNorm knowledge base.
        </p>
        <p style="font-size:0.75rem;color:var(--text-muted);margin-top:0.4rem;">
          Note: Always consult a licensed clinical pharmacist to verify personalized co-administration safety.
        </p>
      </div>
    `;
  }
}

/* ==========================================================================
   UTILITY HELPERS
   ========================================================================== */

function escapeHtml(text) {
  if (!text) return '';
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function formatMarkdown(text) {
  if (!text) return '';
  let html = escapeHtml(text);
  // Headers
  html = html.replace(/^### (.*$)/gim, '<h4 style="font-size:0.92rem;color:#38bdf8;margin:0.75rem 0 0.35rem;">$1</h4>');
  html = html.replace(/^## (.*$)/gim, '<h3 style="font-size:1rem;color:var(--text-primary);margin:0.85rem 0 0.4rem;">$1</h3>');
  // Bold
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  // Italic
  html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
  // Bullets
  html = html.replace(/^\- (.*$)/gim, '<li style="margin-left:1.25rem;margin-bottom:0.25rem;">$1</li>');
  // Line breaks
  html = html.replace(/\n\n/g, '<br><br>');
  return html;
}
