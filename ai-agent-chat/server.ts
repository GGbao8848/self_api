import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import { GoogleGenAI, Type, Content } from "@google/genai";
import { v4 as uuidv4 } from "uuid";

// Initialize Gemini
// Must check if process.env.GEMINI_API_KEY exists
if (!process.env.GEMINI_API_KEY) {
  console.warn("GEMINI_API_KEY is not set in environment.");
}
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || "" });

// Tools Definition
const tools = [
  {
    name: "get_weather",
    description: "Get the weather in a given location",
    parameters: {
      type: Type.OBJECT,
      properties: {
        location: {
          type: Type.STRING,
          description: "The city and state, e.g. San Francisco, CA",
        },
      },
      required: ["location"],
    },
  },
  {
    name: "get_current_time",
    description: "Get the current time in a given timezone",
    parameters: {
      type: Type.OBJECT,
      properties: {
        timezone: {
          type: Type.STRING,
          description: "The timezone, e.g. 'America/Los_Angeles'",
        },
      },
      required: ["timezone"],
    },
  },
];

// Simple tools implementation
async function executeTool(name: string, args: any): Promise<any> {
  if (name === "get_weather") {
    // Mock weather
    const temps = [55, 60, 65, 70, 75, 80];
    const styles = ["Sunny", "Cloudy", "Rainy", "Windy"];
    const t = temps[Math.floor(Math.random() * temps.length)];
    const s = styles[Math.floor(Math.random() * styles.length)];
    return { temperature: `${t}F`, condition: s, location: args.location };
  } else if (name === "get_current_time") {
    try {
      const formatter = new Intl.DateTimeFormat('en-US', {
        timeZone: args.timezone,
        timeStyle: 'long',
        dateStyle: 'full'
      });
      return { time: formatter.format(new Date()), timezone: args.timezone };
    } catch (e: any) {
      return { error: `Invalid timezone: ${args.timezone}` };
    }
  }
  return { error: "Unknown tool" };
}

// Session state (in-memory)
// Keys are session_id, values are arrays of chat entries
interface MessageSession {
  role: "user" | "model";
  text?: string;
  toolCalls?: any[];
  toolResults?: any[];
}
const sessions = new Map<string, MessageSession[]>();

// To store pure Content objects for Gemini history
const geminiSessions = new Map<string, Content[]>();

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // API Routes

  // 1. Tool List Sidebar: /api/v1/agent/tools
  app.get("/api/v1/agent/tools", (req, res) => {
    res.json(tools);
  });

  // 1.5 List all sessions
  app.get("/api/v1/agent/sessions", (req, res) => {
    const sessionList = Array.from(sessions.keys()).map((id) => {
      const history = sessions.get(id) || [];
      const firstUserMessage = history.find(m => m.role === 'user');
      return {
        id,
        messageCount: history.length,
        preview: firstUserMessage?.text || "New Chat"
      };
    }).reverse(); // newest first
    res.json(sessionList);
  });

  // 2. Session History: /api/v1/agent/sessions/:session_id
  app.get("/api/v1/agent/sessions/:session_id", (req, res) => {
    const sid = req.params.session_id;
    const history = sessions.get(sid) || [];
    res.json({ session_id: sid, history });
  });

  // 3. Chat Endpoint: /api/v1/agent/chat
  app.post("/api/v1/agent/chat", async (req, res) => {
    try {
      const { session_id, message } = req.body;
      if (!session_id || !message) {
        return res.status(400).json({ error: "session_id and message are required" });
      }

      if (!sessions.has(session_id)) {
        sessions.set(session_id, []);
        geminiSessions.set(session_id, []);
      }
      const history = sessions.get(session_id)!;
      const geminiHistory = geminiSessions.get(session_id)!;

      // Add user message to history
      history.push({ role: "user", text: message });
      geminiHistory.push({ role: "user", parts: [{ text: message }] });

      // Call Gemini
      let response = await ai.models.generateContent({
        model: "gemini-3.1-pro-preview",
        contents: geminiHistory,
        config: {
          tools: [{ functionDeclarations: tools }],
        },
      });

      // Track the assistant's action
      const newAssistantMessage: MessageSession = { role: "model" };
      let tool_calls_info: any[] = [];

      // Check if there are tool calls
      if (response.functionCalls && response.functionCalls.length > 0) {
        // Prepare to execute tool calls
        const functionCalls = response.functionCalls;

        // Add the model's function calls to the gemini history
        const responseContent = response.candidates?.[0]?.content;
        if (responseContent) {
          geminiHistory.push(responseContent);
        }

        const functionResponses: any[] = [];

        for (const call of functionCalls) {
          const result = await executeTool(call.name, call.args);
          tool_calls_info.push({
            id: call.id,
            name: call.name,
            arguments: call.args,
            result: result.error ? null : result,
            error: result.error || null
          });

          functionResponses.push({
            functionResponse: {
              name: call.name,
              id: call.id,
              response: result,
            }
          });
        }
        newAssistantMessage.toolCalls = tool_calls_info;

        // Add tool responses to gemini history
        geminiHistory.push({ role: "user", parts: functionResponses });

        // Call Gemini again to get final answer
        response = await ai.models.generateContent({
          model: "gemini-3.1-pro-preview",
          contents: geminiHistory,
          config: {
            tools: [{ functionDeclarations: tools }],
          },
        });
      }

      const replyContent = response.candidates?.[0]?.content;
      if (replyContent) {
        geminiHistory.push(replyContent);
      }

      newAssistantMessage.text = response.text || "";
      history.push(newAssistantMessage);

      // Return both text and tool call actions for the frontend
      res.json(newAssistantMessage);

    } catch (error: any) {
      console.error("Chat Error:", error);
      res.status(500).json({ error: error.message || "Internal Server Error" });
    }
  });


  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    // Provide a fallback for React Router
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
