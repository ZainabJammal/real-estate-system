import React, { useState, useRef, useEffect } from "react";
import "./ChatAssistant.css";

const ChatAssistant = () => {
  const initialSystemMessage = {
    role: "system",
    content: "You are a helpful real estate assistant."
  };

  const [messages, setMessages] = useState([initialSystemMessage]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async () => {
    if(!input.trim()) 
      {
        alert("Please enter a message.");
        return;
      }
    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);

    try {
    const res = await fetch("http://localhost:8000/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({messages: [...messages, { role: "user", content: input }]})
    });

    const data = await res.json();
    if (data.error) {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: `Error: ${data.error}` },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: data.reply },
        ]);
      }
    } catch (err) {
      setMessages(prev => [
        ...prev,
        { role: "assistant", content: "Network error, please try again." },
      ]);
    } finally {
      setInput("");
      setLoading(false);
    }
  };

const handleReset = () => {
    setMessages([initialSystemMessage]);
    setInput("");
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>Real Estate AI Assistant</h1>
        <button onClick={handleReset}>Reset Conversation</button>
      </div>

      <div className="chat-messages">
        {messages
          .filter((msg) => msg.role !== "system")
          .map((msg, idx) => (
            <div
              key={idx}
              className={`message ${msg.role === "user" ? "user" : "assistant"}`}
            >
              <div className="bubble">{msg.content}</div>
            </div>
          ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-container">
        <textarea
          rows={2}
          placeholder="Ask anything about real estate..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSubmit()}
        />
        <button onClick={handleSubmit} disabled={loading}>
          {loading ? "Thinking..." : "Send"}
        </button>
      </div>
    </div>
  );
};

export default ChatAssistant;
