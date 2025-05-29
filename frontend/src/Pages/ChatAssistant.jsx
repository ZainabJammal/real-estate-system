import React, { useState, useRef, useEffect } from "react";
import { FaComment, FaPlus } from 'react-icons/fa';
import "./ChatAssistant.css";

const ChatAssistant = () => {
  const REAL_ESTATE_PROMPT = {
    role: "system",
    content: "You are an expert Lebanese real estate assistant specializing in property prices, trends, and recommendations."
  };

  const [sessionId, setSessionId] = useState(() => {
    const saved = localStorage.getItem("chatSessionId");
    if (saved) return saved;
    const newId = crypto.randomUUID();
    localStorage.setItem("chatSessionId", newId);
    return newId;
  });

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  

  // Load chat from Supabase
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await fetch(`http://localhost:8000/api/chat/history/${sessionId}`);
        const data = await res.json();
        setMessages(data.messages.length ? data.messages : [REAL_ESTATE_PROMPT]);
      } catch (err) {
        console.error("History load failed", err);
        setMessages([REAL_ESTATE_PROMPT]); // fallback
      }
    };
    fetchHistory();
}, [sessionId]);


  // Scroll to bottom on new message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async () => {
    if(!input.trim()) return;

    const userMessage = { role: "user", content: input };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput("");
    setLoading(true);

    try {
    const res = await fetch("http://localhost:8000/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: updatedMessages }) 
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Unknown error");

    const finalMessages = [
        ...updatedMessages,
        {
          role: "assistant",
          content: data.reply,
          metadata: data.usage,
        },
      ];
      
    setMessages(finalMessages);

    // Save to Supabase
    await fetch("http://localhost:8000/api/chat/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        messages: finalMessages,
        })
      });
    } catch (err) {
    setMessages((prev) => [
        ...prev,
        { 
          role: "assistant", 
          content: `⚠️ Error: ${err.message}`,
          isError: true
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    };

  const handleClear = () => {
    setMessages([REAL_ESTATE_PROMPT]);
    localStorage.removeItem("chatSessionId");
    const newId = crypto.randomUUID();
    setSessionId(newId);
    localStorage.setItem("chatSessionId", newId);
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h2>
          🏠 Lebanese Real Estate AI
          <button className="new-chat-btn" onClick={handleClear} disabled={messages.length <= 1}>
            <FaComment size={20} color= "white" />
          </button>
        </h2>
      </div>

      <div className="chat-messages">
        {messages
          .filter((msg) => msg.role !== "system")
          .map((msg, idx) => (
            <div key={idx} className={`message ${msg.role} ${msg.isError ? "error" : ""}`}>
              <div className="avatar">{msg.role === "user" ? "🧑" : msg.isError ? "⚠️" : "🤖"}</div>
              <div className="message-content">{msg.content}
                {/* {msg.metadata && (
                  <div className="message-meta">
                    Tokens: {msg.metadata.total_tokens}
                  </div>
                  )} */}
              </div>
            </div>
          ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-area">
        <textarea
          rows={2}
          placeholder="Ask about properties, prices, or trends..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
        />
        <button onClick={handleSubmit} disabled={loading || !input.trim()}>
          {loading ? "⏳ Analyzing..." : "📩 Send"}
        </button>
      </div>
    </div>
  );
};

export default ChatAssistant;
