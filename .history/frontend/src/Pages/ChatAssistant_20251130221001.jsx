import React, { useState, useRef, useEffect } from "react";
import {useSearchParams, useNavigate} from "react-router-dom";
import { FaComment } from "react-icons/fa";
import "./ChatAssistant.css";

const ChatAssistant = () => { 
  const REAL_ESTATE_PROMPT = {
    role: "system",
    content: "You are an expert Lebanese real estate assistant specializing in property prices, trends, and recommendations.",
  };

  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  const [sessionId, setSessionId] = useState(null);

  useEffect(() => {
    const sessionFromUrl = searchParams.get("session");

    if (sessionFromUrl) {
      setSessionId(sessionFromUrl);
    } else {
      // No session in URL → fetch most recent session from Supabase
      const loadLastSession = async () => {
        try {
          const res = await fetch("http://localhost:5000/api/chat/last");
          const data = await res.json();

          if (data.session_id) {
            setSessionId(data.session_id);
            setSearchParams({ session: data.session_id });
          } else {
            // No existing sessions → create new one
            const newId = crypto.randomUUID();
            setSessionId(newId);
            setSearchParams({ session: newId });
          }
        } catch (err) {
          console.error("Failed to load last session", err);
          // Fallback: create new session
          const newId = crypto.randomUUID();
          setSessionId(newId);
          setSearchParams({ session: newId });
        }
      };
      loadLastSession();
    }
  }, []);

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);


  // Load chat history from Supabase
  useEffect(() => {
    if (!sessionId) return; // wait for sessionId to be ready

    const fetchHistory = async () => {
      try {
        const res = await fetch(`http://localhost:5000/api/chat/history/${sessionId}`);
        const data = await res.json();
        setMessages(data.messages.length ? data.messages : [REAL_ESTATE_PROMPT]);
      } catch (err) {
        console.error("History load failed", err);
        setMessages([REAL_ESTATE_PROMPT]);
      }
    };
    
    fetchHistory();
  }, [sessionId]);

  // Scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  //Auto-focus the textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  }, []);


  const handleSubmit = async () => {
    if (!input.trim()) return;

    const userMessage = { role: "user", content: input };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("http://localhost:5000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: updatedMessages }),
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
      await fetch("http://localhost:5000/api/chat/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          messages: finalMessages,
        }),
      });
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `⚠️ Error: ${err.message}`,
          isError: true,
        },
      ]);
    } finally {
      setLoading(false);
      // ⏱ Wait for React to finish re-rendering before focusing
      setTimeout(() => {
        if (textareaRef.current) {
          textareaRef.current.focus();
        }
      }, 0);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleClear = () => {
    const newId = crypto.randomUUID();
    setSessionId(newId);
    setSearchParams({ session: newId });
    setMessages([REAL_ESTATE_PROMPT]);
  };

  const isArabic = (text) => {
    const arabicRegex = /[\u0600-\u06FF]/;
    return arabicRegex.test(text);
  };


  return (
    <div className="chat-container">
      <div className="chat-header">
        <h2>
          🏠 Lebanese Real Estate AI
          <button className="new-chat-btn" onClick={handleClear} disabled={messages.length <= 1}>
            <FaComment size={20} color="white" />
          </button>
        </h2>
      </div>

      <div className="chat-messages">
        {messages
          .filter((msg) => msg.role !== "system")
          .map((msg, idx) => (
            <div key={idx} className={`message ${msg.role} ${msg.isError ? "error" : ""}`}>
              <div className="avatar">{msg.role === "user" ? "🧑" : msg.isError ? "⚠️" : "🤖"}</div>
              <div className="message-content" 
                    dir={isArabic(msg.content) ? "rtl" : "ltr"}
                    style={{ textAlign: isArabic(msg.content) ? "right" : "left" }}>
                {msg.content}
              </div>
            </div>
          ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-area">
        <textarea
          ref={textareaRef}
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
