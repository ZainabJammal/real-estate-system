import React, { useState } from "react";
import "./Page_Layout.css";

const ChatAssistant = () => {
  const [userMessage, setUserMessage] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    setLoading(true);
    setResponse("");
     try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: userMessage }),
    });
    const data = await res.json();
    if (data.error) {
      setResponse(`Error: ${data.error}`);
    } else {
      setResponse(data.reply);
    }
  } catch (err) {
    setResponse("Network error, please try again.");
  }
  setLoading(false);
};  

  return (
    <div className="dashboard-layout">
      <div className="dashboard-content">
        <h1>Ask AI Assistant</h1>
        <textarea
          value={userMessage}
          onChange={(e) => setUserMessage(e.target.value)}
          placeholder="Ask anything about real estate..."
          rows="4"
          className="chat-input"
        />
        <button onClick={handleSubmit} disabled={loading}>
          {loading ? "Thinking..." : "Ask"}
        </button>
        {response && (
          <div className="chat-response">
            <h3>AI Assistant:</h3>
            <p>{response}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatAssistant;
