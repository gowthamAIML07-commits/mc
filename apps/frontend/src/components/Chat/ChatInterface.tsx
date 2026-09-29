import React, { useState } from "react";
import { apiClient } from "../../api/client";
import { ChatResponse, PrescriptionResult } from "../../types";

interface ChatInterfaceProps {
  prescriptionContext: PrescriptionResult | null;
  onClearContext: () => void;
  onOpenCitation: (citationId: string) => void;
}

interface MessageItem {
  sender: "user" | "assistant";
  text: string;
  responseObj?: ChatResponse;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  prescriptionContext,
  onClearContext,
  onOpenCitation
}) => {
  const [messages, setMessages] = useState<MessageItem[]>([
    {
      sender: "assistant",
      text: "Welcome to the Clinical Knowledge RAG Chatbot. Ask any medicine, dosage, or drug interaction question grounded in official DailyMed monographs."
    }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);

  const handleSend = async (textToSend: string) => {
    if (!textToSend.trim()) return;
    const userMsg: MessageItem = { sender: "user", text: textToSend };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const resp = await apiClient.sendChatMessage({
        message: textToSend,
        conversation_id: conversationId,
        prescription_context: prescriptionContext
      });
      setConversationId(resp.conversation_id);
      setMessages((prev) => [
        ...prev,
        { sender: "assistant", text: resp.response, responseObj: resp }
      ]);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        { sender: "assistant", text: `Error: ${err.message || "Failed to generate response."}` }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-workspace">
      {prescriptionContext && (
        <div className="context-ribbon">
          <span>
            <strong>Prescription Context Attached:</strong>{" "}
            {prescriptionContext.candidates.map((c) => c.normalized_name || c.raw_text).join(", ")}
          </span>
          <button className="btn-link btn-xs" onClick={onClearContext}>
            Clear Context
          </button>
        </div>
      )}

      <div className="chat-messages-container">
        {messages.map((m, idx) => (
          <div
            key={idx}
            className={`message-bubble ${m.sender === "user" ? "message-user" : "message-assistant"}`}
          >
            <div className="message-header">
              <span className="author-badge">
                {m.sender === "user" ? "You" : "AI Clinical Pharmacist"}
              </span>
            </div>
            <div className="message-body">{m.text}</div>

            {m.responseObj?.safety_level === "EMERGENCY" && (
              <div className="safety-alert-box safety-alert-emergency" role="alert">
                <strong>🚨 EMERGENCY DIRECTIVE:</strong> Immediate medical attention required. Please call 911/112 or visit the nearest ER.
              </div>
            )}

            {m.responseObj?.citations && m.responseObj.citations.length > 0 && (
              <div className="citations-list">
                {m.responseObj.citations.map((c, cIdx) => (
                  <button
                    key={cIdx}
                    className="citation-pill"
                    onClick={() => onOpenCitation(c.citation_id)}
                  >
                    📄 {c.title} ({c.section})
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="message-bubble message-assistant">
            <span className="time-badge">Consulting DailyMed / RxNorm knowledge base...</span>
          </div>
        )}
      </div>

      <form
        className="chat-composer"
        onSubmit={(e) => {
          e.preventDefault();
          handleSend(input);
        }}
      >
        <input
          type="text"
          className="chat-input"
          placeholder="Ask a medical or prescription question..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <button type="submit" className="btn btn-primary" disabled={loading}>
          Send
        </button>
      </form>
    </div>
  );
};
