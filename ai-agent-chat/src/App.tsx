import React, { type FormEvent, useEffect, useRef, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import {
  Bot,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Loader2,
  MessageSquare,
  PanelLeft,
  PanelRight,
  Plus,
  Send,
  Settings,
  User,
  Wrench,
  XCircle,
} from 'lucide-react';

type AgentRunState =
  | 'completed'
  | 'requires_provider'
  | 'clarification_required'
  | 'failed';

interface ToolDef {
  name: string;
  description: string;
  async_task: boolean;
  argument_hint?: string | null;
}

interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
  result: Record<string, unknown> | null;
  error: string | null;
}

interface AgentRun {
  session_id: string;
  run_id: string;
  user_message?: string | null;
  message: string;
  final_state: AgentRunState;
  provider?: string | null;
  model?: string | null;
  tool_calls: ToolCall[];
}

interface AgentToolsResponse {
  total: number;
  items: ToolDef[];
}

interface AgentSessionResponse {
  session_id: string;
  runs: AgentRun[];
}

interface SessionItem {
  id: string;
  messageCount: number;
  preview: string;
}

interface Message {
  role: 'user' | 'model';
  text: string;
  toolCalls?: ToolCall[];
  state?: AgentRunState;
}

const apiBase = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

function apiUrl(path: string): string {
  return `${apiBase}${path}`;
}

function buildHistoryFromRuns(runs: AgentRun[]): Message[] {
  const messages: Message[] = [];
  for (const run of runs) {
    if (run.user_message) {
      messages.push({
        role: 'user',
        text: run.user_message,
      });
    }
    messages.push({
      role: 'model',
      text: run.message,
      toolCalls: run.tool_calls,
      state: run.final_state,
    });
  }
  return messages;
}

export default function App() {
  const [sessionId, setSessionId] = useState<string>(() => {
    return localStorage.getItem('agent_session_id') || uuidv4();
  });
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [tools, setTools] = useState<ToolDef[]>([]);
  const [history, setHistory] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [leftSidebarOpen, setLeftSidebarOpen] = useState(true);
  const [rightSidebarOpen, setRightSidebarOpen] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const nextScrollBehaviorRef = useRef<ScrollBehavior>('smooth');

  useEffect(() => {
    localStorage.setItem('agent_session_id', sessionId);
    nextScrollBehaviorRef.current = 'auto';
    void fetchHistory(sessionId);
  }, [sessionId]);

  useEffect(() => {
    void fetchTools();
    void fetchSessions();
  }, []);

  useEffect(() => {
    void fetchSessions();
  }, [history]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: nextScrollBehaviorRef.current,
    });
    nextScrollBehaviorRef.current = 'smooth';
  }, [history, loading]);

  const fetchSessions = async () => {
    try {
      const res = await fetch(apiUrl('/api/v1/agent/sessions'), {
        credentials: 'include',
      });
      if (!res.ok) {
        return;
      }
      const data: SessionItem[] = await res.json();
      setSessions(data);
    } catch (error) {
      console.error('Failed to fetch sessions:', error);
    }
  };

  const fetchTools = async () => {
    try {
      const res = await fetch(apiUrl('/api/v1/agent/tools'), {
        credentials: 'include',
      });
      if (!res.ok) {
        return;
      }
      const data: AgentToolsResponse = await res.json();
      setTools(data.items || []);
    } catch (error) {
      console.error('Failed to fetch tools:', error);
    }
  };

  const fetchHistory = async (targetSessionId: string) => {
    try {
      const res = await fetch(apiUrl(`/api/v1/agent/sessions/${targetSessionId}`), {
        credentials: 'include',
      });
      if (res.status === 404) {
        nextScrollBehaviorRef.current = 'auto';
        setHistory([]);
        return;
      }
      if (!res.ok) {
        return;
      }
      const data: AgentSessionResponse = await res.json();
      nextScrollBehaviorRef.current = 'auto';
      setHistory(buildHistoryFromRuns(data.runs || []));
    } catch (error) {
      console.error('Failed to fetch history:', error);
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userText = input.trim();
    setHistory((prev) => [...prev, { role: 'user', text: userText }]);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch(apiUrl('/api/v1/agent/chat'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ session_id: sessionId, message: userText }),
      });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(errorText || `HTTP ${res.status}`);
      }

      const data: AgentRun = await res.json();
      setHistory((prev) => [
        ...prev,
        {
          role: 'model',
          text: data.message,
          toolCalls: data.tool_calls,
          state: data.final_state,
        },
      ]);
      void fetchSessions();
    } catch (error) {
      console.error('Chat error:', error);
      setHistory((prev) => [
        ...prev,
        {
          role: 'model',
          text: 'An error occurred while processing your request.',
          state: 'failed',
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const resetSession = () => {
    const nextSessionId = uuidv4();
    setSessionId(nextSessionId);
    localStorage.setItem('agent_session_id', nextSessionId);
    setHistory([]);
  };

  return (
    <div className="flex h-screen overflow-hidden bg-stone-100 text-stone-900">
      <aside
        className={`${
          leftSidebarOpen ? 'translate-x-0 w-72' : '-translate-x-full w-0'
        } flex-shrink-0 border-r border-stone-200 bg-[#f7f4ee] transition-all duration-300 ease-in-out flex flex-col`}
      >
        <div className="flex h-[69px] items-center justify-between border-b border-stone-200 bg-[#f2eee6] px-4">
          <div className="flex items-center gap-2 font-semibold">
            <MessageSquare className="h-5 w-5 text-teal-700" />
            <span>Sessions</span>
          </div>
          <button
            onClick={() => setLeftSidebarOpen(false)}
            className="text-stone-500 hover:text-stone-700 md:hidden"
          >
            <XCircle className="h-5 w-5" />
          </button>
        </div>

        <div className="flex-1 space-y-3 overflow-y-auto p-4">
          <button
            onClick={resetSession}
            className="flex w-full items-center gap-2 rounded-xl border border-teal-200 bg-teal-50 px-3 py-2 text-sm font-medium text-teal-800 transition-colors hover:bg-teal-100"
          >
            <Plus className="h-4 w-4" />
            New Chat
          </button>

          <div>
            <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-stone-500">
              Recent Sessions
            </div>
            <div className="space-y-1">
              {sessions.map((sess) => (
                <button
                  key={sess.id}
                  onClick={() => setSessionId(sess.id)}
                  className={`flex w-full flex-col gap-1 rounded-xl px-3 py-3 text-left text-sm transition-colors ${
                    sess.id === sessionId
                      ? 'bg-teal-700 text-white shadow-sm'
                      : 'bg-white text-stone-700 hover:bg-stone-50'
                  }`}
                >
                  <span className="block w-full truncate font-medium">{sess.preview}</span>
                  <span className={`text-xs ${sess.id === sessionId ? 'text-teal-100' : 'text-stone-400'}`}>
                    {sess.messageCount} runs
                  </span>
                </button>
              ))}
              {sessions.length === 0 && (
                <div className="rounded-xl bg-white px-3 py-4 text-center text-sm text-stone-500">
                  No sessions yet
                </div>
              )}
            </div>
          </div>
        </div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col bg-stone-100">
        <header className="flex h-[69px] items-center justify-between border-b border-stone-200 bg-white px-4 shadow-sm">
          <div className="flex items-center gap-3">
            {!leftSidebarOpen && (
              <button
                onClick={() => setLeftSidebarOpen(true)}
                className="rounded-md p-1.5 text-stone-500 hover:bg-stone-100"
                title="Toggle sessions"
              >
                <PanelLeft className="h-5 w-5" />
              </button>
            )}
            <h1 className="flex items-center gap-2 text-lg font-semibold">
              <Bot className="h-6 w-6 text-teal-700" />
              Self API Agent
            </h1>
          </div>

          <div className="flex items-center gap-3 text-sm">
            <div className="hidden items-center gap-2 rounded-full border border-stone-200 bg-stone-100 px-3 py-1 sm:flex">
              <span className="h-2 w-2 rounded-full bg-emerald-500" />
              <span className="font-mono text-xs text-stone-600">{sessionId.slice(0, 8)}...</span>
            </div>
            {!rightSidebarOpen && (
              <button
                onClick={() => setRightSidebarOpen(true)}
                className="rounded-md p-1.5 text-stone-500 hover:bg-stone-100"
                title="Toggle capabilities"
              >
                <PanelRight className="h-5 w-5" />
              </button>
            )}
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-4 md:p-6">
          <div className="mx-auto max-w-4xl space-y-6">
            {history.length === 0 && (
              <div className="mt-20 rounded-3xl border border-stone-200 bg-white p-10 text-center shadow-sm">
                <Bot className="mx-auto mb-4 h-16 w-16 text-teal-200" />
                <h2 className="mb-2 text-xl font-medium text-stone-800">Start a tool-driven run</h2>
                <p className="mx-auto max-w-md text-sm text-stone-500">
                  Ask the agent to scan labels, build YAML files, publish datasets, or execute any
                  currently exposed preprocessing tool.
                </p>
              </div>
            )}

            {history.map((msg, idx) => (
              <div
                key={idx}
                className={`flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
              >
                <div
                  className={`flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full ${
                    msg.role === 'user' ? 'bg-teal-700' : 'bg-stone-800'
                  }`}
                >
                  {msg.role === 'user' ? (
                    <User className="h-5 w-5 text-white" />
                  ) : (
                    <Bot className="h-5 w-5 text-white" />
                  )}
                </div>

                <div className={`flex-1 space-y-3 ${msg.role === 'user' ? 'flex flex-col items-end' : ''}`}>
                  <div
                    className={`max-w-[85%] rounded-2xl px-4 py-3 text-[15px] leading-relaxed shadow-sm ${
                      msg.role === 'user'
                        ? 'rounded-tr-sm bg-teal-700 text-white'
                        : 'rounded-tl-sm border border-stone-200 bg-white text-stone-800'
                    }`}
                  >
                    {msg.text}
                  </div>

                  {msg.state && msg.role === 'model' && (
                    <div className="max-w-[85%] text-xs font-medium uppercase tracking-wider text-stone-500">
                      State: {msg.state}
                    </div>
                  )}

                  {msg.toolCalls && msg.toolCalls.length > 0 && (
                    <div className="mt-2 w-full max-w-[85%] space-y-2">
                      {msg.toolCalls.map((call, cidx) => (
                        <ToolCallCard key={cidx} call={call} />
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex gap-4">
                <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-stone-800">
                  <Bot className="h-5 w-5 animate-pulse text-white" />
                </div>
                <div className="flex items-center gap-2 text-sm text-stone-500">
                  <Loader2 className="h-4 w-4 animate-spin text-teal-600" />
                  Thinking and executing...
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        <div className="border-t border-stone-200 bg-white p-4">
          <form onSubmit={handleSubmit} className="mx-auto flex max-w-4xl items-end gap-3">
            <div className="relative flex-1">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    if (!loading && input.trim()) {
                      void handleSubmit(e);
                    }
                  }
                }}
                placeholder="Message the agent. Example: scan label indices for /data/project_a"
                className="max-h-32 min-h-[52px] w-full resize-none rounded-2xl border border-stone-300 bg-stone-50 py-3 pl-4 pr-12 outline-none transition-shadow focus:border-teal-600 focus:ring-1 focus:ring-teal-600"
                rows={1}
                disabled={loading}
              />
              <button
                type="submit"
                disabled={!input.trim() || loading}
                className="absolute bottom-2 right-2 rounded-xl bg-teal-700 p-2 text-white transition-colors hover:bg-teal-800 disabled:opacity-50 disabled:hover:bg-teal-700"
              >
                <Send className="h-4 w-4" />
              </button>
            </div>
          </form>
          <div className="mt-2 text-center text-xs text-stone-400">
            Responses reflect current exposed tools and provider configuration.
          </div>
        </div>
      </main>

      <aside
        className={`${
          rightSidebarOpen ? 'translate-x-0 w-80' : 'translate-x-full w-0'
        } flex-shrink-0 border-l border-stone-200 bg-[#fcfbf8] transition-all duration-300 ease-in-out flex flex-col`}
      >
        <div className="flex h-[69px] items-center justify-between border-b border-stone-200 bg-[#f7f4ee] px-4">
          <div className="flex items-center gap-2 font-semibold">
            <Settings className="h-5 w-5 text-teal-700" />
            <span>Capabilities</span>
          </div>
          <button
            onClick={() => setRightSidebarOpen(false)}
            className="text-stone-500 hover:text-stone-700"
          >
            <XCircle className="h-5 w-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          <h3 className="mb-4 text-xs font-semibold uppercase tracking-wider text-stone-500">
            Available Tools ({tools.length})
          </h3>
          <div className="space-y-4">
            {tools.map((tool) => (
              <div key={tool.name}>
                <ToolSpecCard tool={tool} />
              </div>
            ))}
          </div>
        </div>
      </aside>
    </div>
  );
}

function ToolSpecCard({ tool }: { tool: ToolDef }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="overflow-hidden rounded-2xl border border-stone-200 bg-white">
      <button
        type="button"
        onClick={() => setExpanded((current) => !current)}
        className="flex w-full items-center justify-between gap-3 p-3 text-left transition-colors hover:bg-stone-50"
      >
        <div className="flex min-w-0 items-center gap-2">
          <Wrench className="h-4 w-4 flex-shrink-0 text-teal-600" />
          <div className="min-w-0">
            <h4 className="text-sm font-medium text-stone-900">{tool.name}</h4>
            <p className="mt-1 text-xs text-stone-600">{tool.description}</p>
          </div>
        </div>
        {expanded ? (
          <ChevronDown className="h-4 w-4 flex-shrink-0 text-stone-400" />
        ) : (
          <ChevronRight className="h-4 w-4 flex-shrink-0 text-stone-400" />
        )}
      </button>

      {expanded && (
        <div className="border-t border-stone-100 px-3 py-3 text-[11px]">
          <div className="grid gap-2">
            <div>
              <span className="mb-1 block text-[10px] font-semibold uppercase tracking-wider text-stone-400">
                Async
              </span>
              <div className="rounded-lg bg-stone-50 px-2 py-1 text-stone-700">
                {tool.async_task ? 'true' : 'false'}
              </div>
            </div>
            <div>
              <span className="mb-1 block text-[10px] font-semibold uppercase tracking-wider text-stone-400">
                Argument Hint
              </span>
              <pre className="overflow-x-auto whitespace-pre-wrap rounded-lg bg-stone-50 p-2 text-stone-700">
                {tool.argument_hint || 'No hint'}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ToolCallCard({ call }: { call: ToolCall; key?: React.Key }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="max-w-full overflow-hidden rounded-xl border border-stone-200 bg-white font-mono text-sm shadow-sm">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center justify-between bg-stone-50 p-3 text-left transition-colors hover:bg-stone-100"
      >
        <div className="flex items-center gap-2">
          {call.error ? (
            <XCircle className="h-4 w-4 text-red-500" />
          ) : (
            <CheckCircle2 className="h-4 w-4 text-emerald-500" />
          )}
          <span className="font-semibold text-stone-700">call: {call.name}</span>
          {!expanded && (
            <span className="ml-2 max-w-[180px] truncate text-xs text-stone-500 sm:max-w-xs">
              {JSON.stringify(call.arguments)}
            </span>
          )}
        </div>
        {expanded ? (
          <ChevronDown className="h-4 w-4 text-stone-400" />
        ) : (
          <ChevronRight className="h-4 w-4 text-stone-400" />
        )}
      </button>

      {expanded && (
        <div className="space-y-3 border-t border-stone-100 bg-white p-3">
          <div>
            <span className="mb-1 block text-[10px] font-semibold uppercase tracking-widest text-stone-400">
              Arguments
            </span>
            <pre className="rounded bg-stone-50 p-2 text-xs text-stone-800 whitespace-pre-wrap">
              {JSON.stringify(call.arguments, null, 2)}
            </pre>
          </div>
          <div>
            <span className="mb-1 block text-[10px] font-semibold uppercase tracking-widest text-stone-400">
              Result
            </span>
            {call.error ? (
              <pre className="rounded bg-red-50 p-2 text-xs text-red-600 whitespace-pre-wrap">
                {call.error}
              </pre>
            ) : (
              <pre className="rounded bg-emerald-50 p-2 text-xs text-emerald-700 whitespace-pre-wrap">
                {JSON.stringify(call.result, null, 2)}
              </pre>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
