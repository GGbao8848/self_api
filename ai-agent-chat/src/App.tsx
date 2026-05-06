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
  RotateCcw,
  Send,
  Settings,
  User,
  Wrench,
  XCircle,
} from 'lucide-react';

type AgentRunState =
  | 'accepted'
  | 'running'
  | 'waiting_task'
  | 'interrupted'
  | 'completed'
  | 'requires_provider'
  | 'clarification_required'
  | 'failed'
  | 'cancelled';

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

interface AgentStep {
  step_id: string;
  step_index: number;
  kind: string;
  status: string;
  title: string;
  message?: string | null;
  details?: Record<string, unknown>;
  tool_name?: string | null;
  task_id?: string | null;
  task_type?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  tool_call?: ToolCall | null;
}

interface AgentRun {
  session_id: string;
  run_id: string;
  user_message?: string | null;
  message: string;
  final_state: AgentRunState;
  parent_run_id?: string | null;
  root_run_id?: string | null;
  trigger_kind?: string;
  plan_summary?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  finished_at?: string | null;
  cancellation_requested?: boolean;
  provider?: string | null;
  model?: string | null;
  tool_calls: ToolCall[];
  steps: AgentStep[];
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
  runId?: string;
  parentRunId?: string | null;
  rootRunId?: string | null;
  triggerKind?: string;
  planSummary?: string | null;
  toolCalls?: ToolCall[];
  steps?: AgentStep[];
  state?: AgentRunState;
  canRetry?: boolean;
  canResume?: boolean;
}

interface AgentRunEventPayload {
  event: 'snapshot' | 'end';
  run: AgentRun;
}

const apiBase = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

function apiUrl(path: string): string {
  return `${apiBase}${path}`;
}

function isTerminalState(state?: AgentRunState): boolean {
  return Boolean(
    state &&
    ['interrupted', 'completed', 'requires_provider', 'clarification_required', 'failed', 'cancelled'].includes(state),
  );
}

function isInterruptedState(state?: AgentRunState): boolean {
  return state === 'interrupted';
}

function canRetryState(state?: AgentRunState): boolean {
  return Boolean(
    state &&
    ['completed', 'clarification_required', 'failed', 'cancelled'].includes(state),
  );
}

function buildModelMessage(run: AgentRun): Message {
  return {
    role: 'model',
    runId: run.run_id,
    parentRunId: run.parent_run_id,
    rootRunId: run.root_run_id,
    triggerKind: run.trigger_kind,
    planSummary: run.plan_summary,
    text: run.message,
    toolCalls: run.tool_calls,
    steps: run.steps,
    state: run.final_state,
    canRetry: canRetryState(run.final_state),
    canResume: isInterruptedState(run.final_state),
  };
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
    messages.push(buildModelMessage(run));
  }
  return messages;
}

function getLatestStep(steps?: AgentStep[]): AgentStep | undefined {
  if (!steps || steps.length === 0) {
    return undefined;
  }
  return steps[steps.length - 1];
}

function getLoadingStatus(history: Message[]): { title: string; detail?: string } {
  const activeRun = [...history].reverse().find((item) => item.role === 'model' && item.runId && !isTerminalState(item.state));
  if (!activeRun) {
    return {
      title: 'Working on your request',
      detail: 'Starting the agent run and waiting for the first update.',
    };
  }

  const latestStep = getLatestStep(activeRun.steps);
  if (latestStep?.message) {
    return {
      title: latestStep.title || 'Working on your request',
      detail: latestStep.message,
    };
  }
  if (latestStep?.title) {
    return {
      title: latestStep.title,
      detail: latestStep.tool_name ? `Using ${latestStep.tool_name}.` : undefined,
    };
  }
  if (activeRun.text) {
    return {
      title: 'Working on your request',
      detail: activeRun.text,
    };
  }

  return {
    title: 'Working on your request',
    detail: activeRun.state === 'waiting_task' ? 'Waiting for the current tool to finish.' : 'Planning the next step.',
  };
}

function formatStepStatus(status: string): string {
  switch (status) {
    case 'completed':
      return 'Done';
    case 'running':
      return 'In progress';
    case 'failed':
      return 'Failed';
    case 'cancelled':
      return 'Cancelled';
    default:
      return status;
  }
}

function stepStatusTone(status: string): string {
  switch (status) {
    case 'completed':
      return 'bg-emerald-50 text-emerald-700 border-emerald-200';
    case 'running':
      return 'bg-amber-50 text-amber-700 border-amber-200';
    case 'failed':
      return 'bg-red-50 text-red-700 border-red-200';
    case 'cancelled':
      return 'bg-stone-100 text-stone-600 border-stone-200';
    default:
      return 'bg-stone-100 text-stone-600 border-stone-200';
  }
}

function summarizeStepMeta(step: AgentStep): string[] {
  const items: string[] = [];
  if (step.tool_name) {
    items.push(step.tool_name);
  }
  if (step.kind === 'plan') {
    items.push('planning');
  }
  if (step.task_id) {
    items.push(`task ${step.task_id.slice(0, 8)}`);
  }
  return items;
}

function buildTechnicalDetails(step: AgentStep): string | null {
  const parts: string[] = [];
  if (step.tool_name) {
    parts.push(`Tool: ${step.tool_name}`);
  }
  if (step.task_id) {
    parts.push(`Task ID: ${step.task_id}`);
  }
  if (step.details && Object.keys(step.details).length > 0) {
    parts.push(JSON.stringify(step.details, null, 2));
  }
  return parts.length > 0 ? parts.join('\n\n') : null;
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
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const activeStreamRef = useRef<EventSource | null>(null);
  const loadingStatus = getLoadingStatus(history);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [input]);

  useEffect(() => {
    localStorage.setItem('agent_session_id', sessionId);
    nextScrollBehaviorRef.current = 'auto';
    closeActiveStream();
    void fetchHistory(sessionId);
  }, [sessionId]);

  useEffect(() => {
    void fetchTools();
    void fetchSessions();
    return () => {
      closeActiveStream();
    };
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

  const closeActiveStream = () => {
    if (activeStreamRef.current) {
      activeStreamRef.current.close();
      activeStreamRef.current = null;
    }
  };

  const latestModelRun = (): Message | undefined => {
    return [...history].reverse().find((item) => item.role === 'model' && item.runId);
  };

  const upsertRunMessage = (run: AgentRun) => {
    setHistory((prev) => {
      const next = [...prev];
      const existingIndex = next.findIndex((item) => item.role === 'model' && item.runId === run.run_id);
      const modelMessage = buildModelMessage(run);
      if (existingIndex >= 0) {
        next[existingIndex] = modelMessage;
      } else {
        next.push(modelMessage);
      }
      return next;
    });
  };

  const attachRunStream = (runId: string) => {
    closeActiveStream();
    const eventSource = new EventSource(apiUrl(`/api/v1/agent/runs/${runId}/events`), {
      withCredentials: true,
    });
    activeStreamRef.current = eventSource;

    const handlePayload = (raw: MessageEvent<string>) => {
      const payload: AgentRunEventPayload = JSON.parse(raw.data);
      upsertRunMessage(payload.run);
      if (payload.event === 'end' || isTerminalState(payload.run.final_state)) {
        closeActiveStream();
        setLoading(false);
        void fetchSessions();
      }
    };

    eventSource.addEventListener('snapshot', handlePayload as EventListener);
    eventSource.addEventListener('end', handlePayload as EventListener);
    eventSource.onerror = () => {
      closeActiveStream();
      setLoading(false);
      void fetchHistory(sessionId);
    };
  };

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
      const latestRun = [...(data.runs || [])].reverse().find((run) => !isTerminalState(run.final_state));
      if (latestRun) {
        attachRunStream(latestRun.run_id);
        setLoading(true);
      } else {
        setLoading(false);
      }
    } catch (error) {
      console.error('Failed to fetch history:', error);
    }
  };

  const submitMessage = async (userText: string) => {
    setHistory((prev) => [...prev, { role: 'user', text: userText }]);
    setLoading(true);

    try {
      const priorRun = latestModelRun();
      const requestUrl = priorRun?.runId
        ? apiUrl(`/api/v1/agent/runs/${priorRun.runId}/continue`)
        : apiUrl('/api/v1/agent/chat');
      const requestBody = priorRun?.runId
        ? { message: userText, async_run: true }
        : { session_id: sessionId, message: userText, async_run: true };
      const res = await fetch(requestUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(requestBody),
      });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(errorText || `HTTP ${res.status}`);
      }

      const data: AgentRun = await res.json();
      upsertRunMessage(data);
      if (isTerminalState(data.final_state)) {
        setLoading(false);
      } else {
        attachRunStream(data.run_id);
      }
      void fetchSessions();
    } catch (error) {
      console.error('Chat error:', error);
      setHistory((prev) => [
        ...prev,
        {
          role: 'model',
          text: 'An error occurred while processing your request.',
          state: 'failed',
          canRetry: false,
        },
      ]);
      setLoading(false);
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userText = input.trim();
    setInput('');
    await submitMessage(userText);
  };

  const handleRetryRun = async (runId: string) => {
    setLoading(true);
    try {
      const res = await fetch(apiUrl(`/api/v1/agent/runs/${runId}/retry`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ async_run: true }),
      });
      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(errorText || `HTTP ${res.status}`);
      }
      const data: AgentRun = await res.json();
      upsertRunMessage(data);
      if (isTerminalState(data.final_state)) {
        setLoading(false);
      } else {
        attachRunStream(data.run_id);
      }
      void fetchSessions();
    } catch (error) {
      console.error('Retry error:', error);
      setLoading(false);
    }
  };

  const handleContinueFromRun = async (runId: string) => {
    const userText = input.trim();
    if (!userText || loading) return;
    setInput('');
    setHistory((prev) => [...prev, { role: 'user', text: userText }]);
    setLoading(true);
    try {
      const res = await fetch(apiUrl(`/api/v1/agent/runs/${runId}/continue`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ message: userText, async_run: true }),
      });
      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(errorText || `HTTP ${res.status}`);
      }
      const data: AgentRun = await res.json();
      upsertRunMessage(data);
      if (isTerminalState(data.final_state)) {
        setLoading(false);
      } else {
        attachRunStream(data.run_id);
      }
      void fetchSessions();
    } catch (error) {
      console.error('Continue error:', error);
      setLoading(false);
    }
  };

  const handleResumeRun = async (runId: string) => {
    setLoading(true);
    try {
      const res = await fetch(apiUrl(`/api/v1/agent/runs/${runId}/resume`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({}),
      });
      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(errorText || `HTTP ${res.status}`);
      }
      const data: AgentRun = await res.json();
      upsertRunMessage(data);
      if (isTerminalState(data.final_state) || isInterruptedState(data.final_state)) {
        setLoading(false);
      } else {
        attachRunStream(data.run_id);
      }
      void fetchSessions();
    } catch (error) {
      console.error('Resume error:', error);
      setLoading(false);
    }
  };

  const resetSession = () => {
    const nextSessionId = uuidv4();
    closeActiveStream();
    setSessionId(nextSessionId);
    localStorage.setItem('agent_session_id', nextSessionId);
    setHistory([]);
    setLoading(false);
  };

  return (
    <div className="flex h-screen overflow-hidden bg-stone-100 text-stone-900">
      <aside
        className={`${leftSidebarOpen ? 'translate-x-0 w-72' : '-translate-x-full w-0'
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
            <div className="min-w-0 space-y-1">
              {sessions.map((sess) => (
                <button
                  key={sess.id}
                  onClick={() => setSessionId(sess.id)}
                  className={`flex w-full min-w-0 flex-col gap-1 rounded-xl px-3 py-3 text-left text-sm transition-colors ${sess.id === sessionId
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
              <span className={`h-2 w-2 rounded-full ${loading ? 'bg-amber-500' : 'bg-emerald-500'}`} />
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
                <h2 className="mb-2 text-xl font-medium text-stone-800">Start a long-running agent run</h2>
                <p className="mx-auto max-w-md text-sm text-stone-500">
                  The agent now runs in background mode by default, streams step updates, and can be retried
                  when a long task stalls or fails.
                </p>
              </div>
            )}

            {history.map((msg, idx) => (
              <div
                key={`${msg.role}-${msg.runId || idx}`}
                className={`flex min-w-0 gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
              >
                <div
                  className={`flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full ${msg.role === 'user' ? 'bg-teal-700' : 'bg-stone-800'
                    }`}
                >
                  {msg.role === 'user' ? (
                    <User className="h-5 w-5 text-white" />
                  ) : (
                    <Bot className="h-5 w-5 text-white" />
                  )}
                </div>

                <div className={`min-w-0 flex-1 space-y-3 ${msg.role === 'user' ? 'flex flex-col items-end' : ''}`}>
                  <div
                    className={`max-w-[85%] whitespace-pre-wrap break-words rounded-2xl px-4 py-3 text-[15px] leading-relaxed shadow-sm [overflow-wrap:anywhere] ${msg.role === 'user'
                        ? 'rounded-tr-sm bg-teal-700 text-white'
                        : 'rounded-tl-sm border border-stone-200 bg-white text-stone-800'
                      }`}
                  >
                    {msg.text}
                  </div>

                  {msg.state && msg.role === 'model' && (
                    <div className="flex max-w-[85%] min-w-0 flex-wrap items-center gap-2 text-xs font-medium uppercase tracking-wider text-stone-500">
                      <span>State: {msg.state}</span>
                      {msg.runId && (
                        <span className="rounded-full bg-stone-200 px-2 py-0.5 font-mono normal-case text-stone-700">
                          {msg.runId.slice(0, 8)}
                        </span>
                      )}
                      {msg.triggerKind && (
                        <span className="rounded-full bg-amber-50 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-amber-700">
                          {msg.triggerKind}
                        </span>
                      )}
                      {msg.canRetry && msg.runId && (
                        <button
                          type="button"
                          onClick={() => void handleRetryRun(msg.runId!)}
                          className="inline-flex items-center gap-1 rounded-full border border-stone-300 bg-white px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-stone-700 hover:bg-stone-50"
                        >
                          <RotateCcw className="h-3 w-3" />
                          Retry
                        </button>
                      )}
                      {msg.canResume && msg.runId && (
                        <button
                          type="button"
                          onClick={() => void handleResumeRun(msg.runId!)}
                          className="inline-flex items-center gap-1 rounded-full border border-amber-300 bg-amber-50 px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-amber-700 hover:bg-amber-100"
                        >
                          <RotateCcw className="h-3 w-3" />
                          Resume
                        </button>
                      )}
                      {msg.runId && isTerminalState(msg.state) && (
                        <button
                          type="button"
                          onClick={() => void handleContinueFromRun(msg.runId!)}
                          className="inline-flex items-center gap-1 rounded-full border border-teal-300 bg-teal-50 px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-teal-700 hover:bg-teal-100"
                        >
                          <Send className="h-3 w-3" />
                          Continue
                        </button>
                      )}
                    </div>
                  )}

                  {msg.steps && msg.steps.length > 0 && (
                    <div className="w-full max-w-[85%] min-w-0">
                      <StepTimeline steps={msg.steps} />
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
                <div className="rounded-2xl border border-stone-200 bg-white px-4 py-3 shadow-sm">
                  <div className="flex items-center gap-2 text-sm font-medium text-stone-800">
                    <Loader2 className="h-4 w-4 animate-spin text-teal-600" />
                    <span>{loadingStatus.title}</span>
                  </div>
                  {loadingStatus.detail && (
                    <div className="mt-1 text-sm text-stone-500">{loadingStatus.detail}</div>
                  )}
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
                ref={textareaRef}
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
                placeholder="Message the agent. Example: 先清洗嵌套目录，再转 yolo，然后检查标签索引"
                className="max-h-32 min-h-[52px] w-full resize-none rounded-2xl border border-stone-300 bg-stone-50 py-3 pl-4 pr-12 outline-none transition-shadow focus:border-teal-600 focus:ring-1 focus:ring-teal-600"
                rows={1}
                disabled={loading}
              />
              <button
                type="submit"
                disabled={!input.trim() || loading}
                className="absolute top-1/2 right-2 -translate-y-1/2 rounded-xl bg-teal-700 p-2 text-white transition-colors hover:bg-teal-800 disabled:opacity-50 disabled:hover:bg-teal-700"
              >
                <Send className="h-4 w-4" />
              </button>
            </div>
          </form>
          <div className="mt-2 text-center text-xs text-stone-400">
            Default mode is long-running background execution with live step updates.
          </div>
        </div>
      </main>

      <aside
        className={`${rightSidebarOpen ? 'translate-x-0 w-80' : 'translate-x-full w-0'
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
          <div className="mb-4 rounded-2xl border border-stone-200 bg-white p-4 text-sm text-stone-700 shadow-sm">
            <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-stone-500">
              Experience Upgrades
            </div>
            <ul className="space-y-2 text-[13px] leading-relaxed">
              <li>Background long-task execution</li>
              <li>Live step timeline</li>
              <li>Retry from prior run context</li>
              <li>Resume after service restart</li>
            </ul>
          </div>
          <h3 className="mb-4 text-xs font-semibold uppercase tracking-wider text-stone-500">
            Available Tools ({tools.length})
          </h3>
          <div className="min-w-0 space-y-4">
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
    <div className="min-w-0 overflow-hidden rounded-2xl border border-stone-200 bg-white">
      <button
        type="button"
        onClick={() => setExpanded((current) => !current)}
        className="flex w-full min-w-0 items-center justify-between gap-3 p-3 text-left transition-colors hover:bg-stone-50"
      >
        <div className="flex min-w-0 items-center gap-2">
          <Wrench className="h-4 w-4 flex-shrink-0 text-teal-600" />
          <div className="min-w-0">
            <h4 className="break-words text-sm font-medium text-stone-900 [overflow-wrap:anywhere]">{tool.name}</h4>
            <p className="mt-1 break-words text-xs text-stone-600 [overflow-wrap:anywhere]">{tool.description}</p>
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
              <pre className="max-w-full whitespace-pre-wrap break-words rounded-lg bg-stone-50 p-2 text-stone-700 [overflow-wrap:anywhere]">
                {tool.argument_hint || 'No hint'}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function StepTimeline({ steps }: { steps: AgentStep[] }) {
  if (!steps || steps.length === 0) return null;
  return (
    <div className="mt-2 w-full space-y-2">
      <div className="mb-2 px-1 text-xs font-semibold uppercase tracking-wider text-stone-500">
        Agent Action Sequence
      </div>
      {steps.map((step) => (
        <StepCard key={step.step_id} step={step} />
      ))}
    </div>
  );
}

function formatStatusText(status: string) {
  if (status === 'running') return 'Running...';
  if (status === 'completed') return 'Success';
  if (status === 'failed') return 'Failed';
  return formatStepStatus(status);
}

function StepCard({ step }: { step: AgentStep; key?: React.Key }) {
  const [expanded, setExpanded] = useState(false);

  let headerLabel = step.title || `Step ${step.step_index}`;
  if (step.kind === 'tool_call' || step.tool_name) {
    headerLabel = `call: ${step.tool_name || step.tool_call?.name || step.title}`;
  } else if (step.kind === 'plan' || step.title.toLowerCase().includes('plan')) {
    headerLabel = `plan: ${step.message || step.title}`;
  } else {
    headerLabel = `step ${step.step_index}: ${step.title}`;
  }

  return (
    <div
      className={`max-w-full min-w-0 overflow-hidden rounded-xl border border-stone-200 bg-white font-mono text-sm shadow-sm ${
        step.status === 'failed' ? 'border-red-200' : ''
      }`}
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full min-w-0 items-center justify-between gap-3 bg-stone-50 p-3 text-left transition-colors hover:bg-stone-100"
      >
        <div className="flex min-w-0 items-center gap-2">
          {step.status === 'failed' ? (
            <XCircle className="h-4 w-4 text-red-500" />
          ) : step.status === 'completed' ? (
            <CheckCircle2 className="h-4 w-4 text-emerald-500" />
          ) : (
            <Loader2 className="h-4 w-4 animate-spin text-amber-500" />
          )}
          <span className="truncate font-semibold text-stone-700">{headerLabel}</span>
          {!expanded && (
            <span className="ml-2 max-w-[150px] truncate text-xs text-stone-500 sm:max-w-xs">
              {formatStatusText(step.status)}
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
        <div className="flex w-full min-w-0 flex-col items-start space-y-3 border-t border-stone-100 bg-white p-3 text-left">
          {step.message && (
            <div className="w-full">
              <span className="mb-1 block text-[10px] font-semibold uppercase tracking-widest text-stone-400">
                Message
              </span>
              <div className="rounded bg-stone-50 p-2 text-xs whitespace-pre-wrap break-words text-stone-800 [overflow-wrap:anywhere]">{step.message}</div>
            </div>
          )}

          {step.tool_call?.arguments && (
            <div className="w-full">
              <span className="mb-1 block text-[10px] font-semibold uppercase tracking-widest text-stone-400">
                Arguments
              </span>
              <pre className="max-w-full whitespace-pre-wrap break-words rounded bg-stone-50 p-2 text-xs text-stone-800 [overflow-wrap:anywhere]">
                {JSON.stringify(step.tool_call.arguments, null, 2)}
              </pre>
            </div>
          )}

          {step.tool_call?.result && (
            <div className="w-full">
              <span className="mb-1 block text-[10px] font-semibold uppercase tracking-widest text-stone-400">
                Result
              </span>
              <pre className="max-w-full whitespace-pre-wrap break-words rounded bg-emerald-50 p-2 text-xs text-emerald-700 [overflow-wrap:anywhere]">
                {JSON.stringify(step.tool_call.result, null, 2)}
              </pre>
            </div>
          )}

          {step.tool_call?.error && (
            <div className="w-full">
              <span className="mb-1 block text-[10px] font-semibold uppercase tracking-widest text-stone-400">
                Error
              </span>
              <pre className="max-w-full whitespace-pre-wrap break-words rounded bg-red-50 p-2 text-xs text-red-600 [overflow-wrap:anywhere]">
                {step.tool_call.error}
              </pre>
            </div>
          )}

          {!step.tool_call && step.details && Object.keys(step.details).length > 0 && (
            <div className="w-full">
              <span className="mb-1 block text-[10px] font-semibold uppercase tracking-widest text-stone-400">
                Details
              </span>
              <pre className="max-w-full whitespace-pre-wrap break-words rounded bg-stone-50 p-2 text-xs text-stone-800 [overflow-wrap:anywhere]">
                {JSON.stringify(step.details, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
