import React, { useState, useRef, useEffect } from "react";
import { FaComment, FaPlus } from 'react-icons/fa';
import "./ChatAssistant.css";

const ChatAssistant = () => {
  const REAL_ESTATE_PROMPT = {
    role: "system",
    content: "You are an expert Lebanese real estate assistant specializing in property prices, trends, and recommendations."
  };

  const [messages, setMessages] = useState([REAL_ESTATE_PROMPT]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Load from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem("chatHistory");
    if (saved) {
      setMessages(JSON.parse(saved));
    }
  }, []);

  // Save to localStorage whenever messages change
  useEffect(() => {
    localStorage.setItem("chatHistory", JSON.stringify(messages));
  }, [messages]);

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
      body: JSON.stringify({ messages: updatedMessages }) // Include full history
    });

    const data = await res.json();
    
    if (!res.ok) throw new Error(data.error || "Unknown error");
      
    setMessages((prev) => [
      ...prev,
      { 
        role: "assistant", 
        content: data.reply,
        metadata: data.usage,  // Store token usage if needed
      },
    ]);
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
    localStorage.removeItem("chatHistory");
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
            <div
              key={idx}
              className={`message ${msg.role} ${msg.isError ? "error" : ""}`}>
                
              <div className="avatar">
                {msg.role === "user" ? "🧑" : msg.isError ? "⚠️" : "🤖"}
              </div>
              <div className="message-content">
                {msg.content}
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
