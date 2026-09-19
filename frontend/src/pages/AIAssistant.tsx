import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { fetchMachineHealth, sendAssistantMessage } from "../api/client";
import type { AssistantResponse } from "../types/assistant";
import type {
  HealthStatus,
  MachineHealthResponse,
} from "../types/machineHealth";

type MessageRole = "user" | "assistant";

interface CopilotMessage {
  id: string;
  role: MessageRole;
  content: string;
  createdAt: string;
}

interface CopilotThread {
  id: string;
  title: string;
  udi: number;
  status: HealthStatus | "UNKNOWN";
  createdAt: string;
  updatedAt: string;
  messages: CopilotMessage[];
}

const STORAGE_KEY = "industrial-machine-health-copilot-threads";

const QUICK_PROMPTS = [
  "Explain this machine status in simple language.",
  "What should the maintenance team inspect first?",
  "Which sensor signals are contributing most?",
  "Can normal operation continue?",
];

function createId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function createThread(udi = 7012): CopilotThread {
  const now = new Date().toISOString();

  return {
    id: createId(),
    title: `UDI ${udi} — New investigation`,
    udi,
    status: "UNKNOWN",
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

function readThreads(): CopilotThread[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);

    if (!raw) {
      return [createThread(7012)];
    }

    const parsed = JSON.parse(raw);

    if (!Array.isArray(parsed) || parsed.length === 0) {
      return [createThread(7012)];
    }

    return parsed;
  } catch {
    return [createThread(7012)];
  }
}

function saveThreads(threads: CopilotThread[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(threads));
}

function cleanAssistantText(text: string): string {
  return text
    .replace(/\*\*/g, "")
    .replace(/#{1,6}\s*/g, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function detectStatus(text: string): HealthStatus | "UNKNOWN" {
  const value = text.toUpperCase();

  if (value.includes("CRITICAL")) return "CRITICAL";
  if (value.includes("ALERT")) return "ALERT";
  if (value.includes("WATCH")) return "WATCH";
  if (value.includes("NORMAL")) return "NORMAL";

  return "UNKNOWN";
}

function statusClass(status: HealthStatus | "UNKNOWN"): string {
  return `copilot-status-${status.toLowerCase()}`;
}

function statusDescription(status: HealthStatus | "UNKNOWN"): string {
  switch (status) {
    case "CRITICAL":
      return "The combined sensor pattern is far outside the healthy baseline. Priority investigation is recommended.";
    case "ALERT":
      return "The machine has a meaningful multivariate deviation. Plan a maintenance review soon.";
    case "WATCH":
      return "An early deviation is present. Increase monitoring and inspect during the next scheduled check.";
    case "NORMAL":
      return "The machine remains within the established healthy monitoring boundary.";
    default:
      return "Load a machine to view the verified monitoring assessment.";
  }
}

function statusAction(
  status: HealthStatus | "UNKNOWN",
  recommendation?: string
): string {
  if (recommendation) {
    return recommendation;
  }

  switch (status) {
    case "CRITICAL":
      return "Verify operating conditions, inspect the contributing sensors, and arrange priority maintenance review.";
    case "ALERT":
      return "Schedule a maintenance investigation and compare the machine with normal operating conditions.";
    case "WATCH":
      return "Monitor readings more closely and verify sensor conditions during the next inspection.";
    case "NORMAL":
      return "Continue routine operation and preventive maintenance procedures.";
    default:
      return "Select a valid UDI and load its machine-health evidence.";
  }
}

function shortThreadTime(value: string): string {
  const date = new Date(value);
  const now = new Date();
  const isToday = date.toDateString() === now.toDateString();

  if (isToday) {
    return date.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  return date.toLocaleDateString([], {
    day: "2-digit",
    month: "short",
  });
}

function formatMetric(value: number | undefined, digits = 2): string {
  const numberValue = Number(value);

  return Number.isFinite(numberValue) ? numberValue.toFixed(digits) : "—";
}

function sensorName(value: string): string {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

export default function AIAssistant() {
  const navigate = useNavigate();

  const [threads, setThreads] = useState<CopilotThread[]>(readThreads);
  const [activeThreadId, setActiveThreadId] = useState(() => {
    const initialThreads = readThreads();
    return initialThreads[0]?.id ?? "";
  });
  const [udiInput, setUdiInput] = useState("7012");
  const [machine, setMachine] = useState<MachineHealthResponse | null>(null);
  const [machineLoading, setMachineLoading] = useState(false);
  const [machineError, setMachineError] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [sending, setSending] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);

  const activeThread = useMemo(
    () => threads.find((thread) => thread.id === activeThreadId) ?? threads[0],
    [activeThreadId, threads]
  );

  useEffect(() => {
    saveThreads(threads);
  }, [threads]);

  useEffect(() => {
    if (!activeThread) {
      return;
    }

    setUdiInput(String(activeThread.udi));
  }, [activeThread?.id]);

  useEffect(() => {
    if (!activeThread) {
      return;
    }

    let active = true;

    async function loadMachine() {
      setMachineLoading(true);
      setMachineError(null);

      try {
        const response = await fetchMachineHealth(activeThread.udi);

        if (!active) {
          return;
        }

        setMachine(response);

        setThreads((current) =>
          current.map((thread) =>
            thread.id === activeThread.id
              ? {
                  ...thread,
                  status: response.health_status,
                  title:
                    thread.messages.length === 0
                      ? `UDI ${response.udi} — ${response.health_status.toLowerCase()} review`
                      : thread.title,
                  updatedAt: new Date().toISOString(),
                }
              : thread
          )
        );
      } catch (requestError: unknown) {
        if (!active) {
          return;
        }

        const axiosError = requestError as {
          response?: { status?: number; data?: { detail?: string } };
          message?: string;
        };

        setMachine(null);

        if (axiosError.response?.status === 404) {
          setMachineError(
            `UDI ${activeThread.udi} was not found in the Phase-II monitoring dataset.`
          );
        } else {
          setMachineError(
            axiosError.response?.data?.detail ??
              axiosError.message ??
              "Unable to load machine evidence."
          );
        }
      } finally {
        if (active) {
          setMachineLoading(false);
        }
      }
    }

    loadMachine();

    return () => {
      active = false;
    };
  }, [activeThread?.id, activeThread?.udi]);

  function updateThread(
    threadId: string,
    updater: (thread: CopilotThread) => CopilotThread
  ) {
    setThreads((current) =>
      current.map((thread) => (thread.id === threadId ? updater(thread) : thread))
    );
  }

  function startNewThread() {
    const newThread = createThread(7012);

    setThreads((current) => [newThread, ...current]);
    setActiveThreadId(newThread.id);
    setQuestion("");
    setChatError(null);
  }

  function deleteActiveThread() {
    if (!activeThread) {
      return;
    }

    const remaining = threads.filter((thread) => thread.id !== activeThread.id);

    if (remaining.length === 0) {
      const replacement = createThread(7012);
      setThreads([replacement]);
      setActiveThreadId(replacement.id);
      return;
    }

    setThreads(remaining);
    setActiveThreadId(remaining[0].id);
  }

  function loadSelectedMachine() {
    if (!activeThread) {
      return;
    }

    const parsedUdi = Number(udiInput);

    if (!Number.isInteger(parsedUdi) || parsedUdi <= 0) {
      setMachineError("Enter a valid positive machine UDI.");
      return;
    }

    setMachineError(null);
    setChatError("");

    updateThread(activeThread.id, (thread) => ({
      ...thread,
      udi: parsedUdi,
      title: `UDI ${parsedUdi} — New investigation`,
      status: "UNKNOWN",
      messages: [],
      updatedAt: new Date().toISOString(),
    }));
  }

  async function askCopilot(prompt?: string) {
    if (!activeThread || sending) {
      return;
    }

    const finalQuestion = (prompt ?? question).trim();

    if (!finalQuestion) {
      setChatError("Type a question before sending it to the copilot.");
      return;
    }

    const userMessage: CopilotMessage = {
      id: createId(),
      role: "user",
      content: finalQuestion,
      createdAt: new Date().toISOString(),
    };

    updateThread(activeThread.id, (thread) => ({
      ...thread,
      messages: [...thread.messages, userMessage],
      updatedAt: userMessage.createdAt,
      title:
        thread.messages.length === 0
          ? `UDI ${thread.udi} — ${finalQuestion.slice(0, 34)}`
          : thread.title,
    }));

    setQuestion("");
    setChatError(null);
    setSending(true);

    try {
      const response: AssistantResponse = await sendAssistantMessage(
        activeThread.udi,
        finalQuestion
      );

      const responseText = cleanAssistantText(response.response ?? "");

      const assistantMessage: CopilotMessage = {
        id: createId(),
        role: "assistant",
        content:
          responseText ||
          "The copilot returned no readable response for this question.",
        createdAt: new Date().toISOString(),
      };

      updateThread(activeThread.id, (thread) => ({
        ...thread,
        messages: [...thread.messages, assistantMessage],
        status:
          machine?.health_status ??
          detectStatus(responseText) ??
          thread.status,
        updatedAt: assistantMessage.createdAt,
      }));
    } catch (requestError: unknown) {
      const axiosError = requestError as {
        response?: { status?: number; data?: { detail?: string } };
        message?: string;
      };

      setChatError(
        axiosError.response?.data?.detail ??
          axiosError.message ??
          "The Maintenance Copilot could not respond."
      );
    } finally {
      setSending(false);
    }
  }

  const displayStatus = machine?.health_status ?? activeThread?.status ?? "UNKNOWN";
  const sortedSensors = [...(machine?.contributing_sensors ?? [])].sort(
    (first, second) => Math.abs(second.z_score) - Math.abs(first.z_score)
  );

  return (
    <div className="copilot-page">
      <div className="copilot-app">
        <aside className="copilot-threads-panel">
          <div className="copilot-threads-header">
            <div>
              <div className="copilot-eyebrow">Workspace</div>
              <h1>Investigations</h1>
            </div>

            <button
              type="button"
              className="copilot-new-thread"
              onClick={startNewThread}
              title="Start a new investigation"
            >
              + New
            </button>
          </div>

          <p className="copilot-threads-caption">
            Saved maintenance conversations on this device.
          </p>

          <div className="copilot-thread-list">
            {threads.map((thread) => (
              <button
                type="button"
                className={`copilot-thread-item ${
                  thread.id === activeThread?.id ? "active" : ""
                }`}
                key={thread.id}
                onClick={() => {
                  setActiveThreadId(thread.id);
                  setQuestion("");
                  setChatError(null);
                }}
              >
                <span className={`copilot-thread-status ${statusClass(thread.status)}`} />
                <span className="copilot-thread-copy">
                  <strong>{thread.title}</strong>
                  <small>
                    UDI {thread.udi} · {shortThreadTime(thread.updatedAt)}
                  </small>
                </span>
              </button>
            ))}
          </div>

          <div className="copilot-threads-footer">
            <span className="copilot-live-dot" />
            Read-only statistical decision support
          </div>
        </aside>

        <section className="copilot-conversation-panel">
          <header className="copilot-header">
            <div>
              <div className="copilot-eyebrow">
                AI maintenance copilot · evidence connected
              </div>
              <h2>Machine investigation</h2>
              <p>
                UDI {activeThread?.udi ?? "—"} · Phase-II monitoring · Healthy
                Phase-I baseline
              </p>
            </div>

            <button
              type="button"
              className="copilot-delete-thread"
              onClick={deleteActiveThread}
              title="Delete current investigation"
            >
              Delete thread
            </button>
          </header>

          <section
            className={`copilot-status-panel ${statusClass(displayStatus)}`}
          >
            <div className="copilot-status-main">
              <div className="copilot-status-label">Current assessment</div>
              <div className="copilot-status-value">{displayStatus}</div>
              <p>{statusDescription(displayStatus)}</p>
            </div>

            <div className="copilot-stat-strip">
              <div>
                <span>Hotelling’s T²</span>
                <strong>{formatMetric(machine?.t2_value, 4)}</strong>
              </div>

              <div>
                <span>Control limit</span>
                <strong>{formatMetric(machine?.ucl, 4)}</strong>
              </div>

              <div>
                <span>T² / UCL</span>
                <strong>
                  {machine ? `${formatMetric(machine.t2_ratio, 2)}×` : "—"}
                </strong>
              </div>
            </div>
          </section>

          <section className="copilot-recommendation">
            <div className="copilot-recommendation-icon">→</div>
            <div>
              <span>Recommended next action</span>
              <strong>
                {statusAction(displayStatus, machine?.primary_recommendation)}
              </strong>
            </div>
          </section>

          <div className="copilot-chat">
            <div className="copilot-chat-heading">
              <div>
                <h3>Conversation</h3>
                <p>Ask follow-up questions about the selected machine.</p>
              </div>

              {sending ? (
                <span className="copilot-thinking">Reviewing evidence…</span>
              ) : null}
            </div>

            <div className="copilot-message-list">
              {!activeThread?.messages.length ? (
                <div className="copilot-empty-state">
                  <div className="copilot-empty-mark">AI</div>
                  <h3>Start the investigation</h3>
                  <p>
                    Ask what the result means, which sensor patterns matter,
                    or what to inspect first.
                  </p>

                  <div className="copilot-prompt-chips">
                    {QUICK_PROMPTS.map((prompt) => (
                      <button
                        type="button"
                        key={prompt}
                        disabled={sending}
                        onClick={() => askCopilot(prompt)}
                      >
                        {prompt}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                activeThread.messages.map((message) => (
                  <article
                    className={`copilot-message copilot-message-${message.role}`}
                    key={message.id}
                  >
                    <div className="copilot-message-meta">
                      {message.role === "assistant"
                        ? "Maintenance Copilot"
                        : "You"}
                      <span>{shortThreadTime(message.createdAt)}</span>
                    </div>

                    <div className="copilot-message-content">
                      {message.content}
                    </div>
                  </article>
                ))
              )}

              {sending ? (
                <article className="copilot-message copilot-message-assistant">
                  <div className="copilot-message-meta">
                    Maintenance Copilot
                  </div>
                  <div className="copilot-typing">
                    <span />
                    <span />
                    <span />
                    Reviewing verified machine evidence
                  </div>
                </article>
              ) : null}
            </div>

            {chatError ? (
              <div className="copilot-error">{chatError}</div>
            ) : null}

            <form
              className="copilot-composer"
              onSubmit={(event) => {
                event.preventDefault();
                askCopilot();
              }}
            >
              <textarea
                rows={2}
                value={question}
                disabled={sending}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Ask about this machine’s health, evidence, contributing sensors, or maintenance checks…"
              />

              <button
                type="submit"
                disabled={sending || !question.trim()}
                aria-label="Send message"
              >
                {sending ? "…" : "Send"}
              </button>
            </form>

            <p className="copilot-limit-note">
              The copilot interprets the platform’s statistical outputs. A
              high T² value indicates an abnormal sensor pattern; it does not
              alone confirm a physical fault or a calibrated failure
              probability.
            </p>
          </div>
        </section>

        <aside className="copilot-evidence-panel">
          <div className="copilot-evidence-heading">
            <div>
              <div className="copilot-eyebrow">Live context</div>
              <h2>Machine evidence</h2>
            </div>

            <span className={`copilot-status-pill ${statusClass(displayStatus)}`}>
              {displayStatus}
            </span>
          </div>

          <label className="copilot-udi-label" htmlFor="copilot-udi">
            Machine UDI
          </label>

          <div className="copilot-udi-row">
            <input
              id="copilot-udi"
              type="number"
              min="1"
              value={udiInput}
              disabled={machineLoading || sending}
              onChange={(event) => setUdiInput(event.target.value)}
            />

            <button
              type="button"
              onClick={loadSelectedMachine}
              disabled={machineLoading || sending}
            >
              {machineLoading ? "Loading" : "Load"}
            </button>
          </div>

          {machineError ? (
            <div className="copilot-evidence-error">{machineError}</div>
          ) : null}

          <div className="copilot-evidence-section">
            <h3>Top contributing signals</h3>

            {machineLoading ? (
              <p className="copilot-muted">Loading machine evidence…</p>
            ) : sortedSensors.length ? (
              <div className="copilot-signal-list">
                {sortedSensors.slice(0, 4).map((sensor) => (
                  <div
                    className="copilot-signal"
                    key={`${sensor.sensor}-${sensor.direction}`}
                  >
                    <div>
                      <strong>{sensorName(sensor.sensor)}</strong>
                      <span>
                        {sensor.direction === "HIGH"
                          ? "Above healthy baseline"
                          : "Below healthy baseline"}
                      </span>
                    </div>

                    <b
                      className={
                        sensor.direction === "HIGH"
                          ? "copilot-signal-high"
                          : "copilot-signal-low"
                      }
                    >
                      {sensor.z_score > 0 ? "+" : ""}
                      {formatMetric(sensor.z_score, 2)}
                    </b>
                  </div>
                ))}
              </div>
            ) : (
              <p className="copilot-muted">
                No individual sensor contributors are available for this
                machine.
              </p>
            )}
          </div>

          <div className="copilot-evidence-section">
            <h3>Evidence source</h3>
            <div className="copilot-source-card">
              <span className="copilot-source-dot" />
              <div>
                <strong>Phase-II Hotelling’s T² monitor</strong>
                <p>
                  Compared with the established healthy Phase-I baseline.
                </p>
              </div>
            </div>
          </div>

          <button
            type="button"
            className="copilot-open-report"
            onClick={() =>
              navigate(`/machine/${activeThread?.udi ?? 7012}`)
            }
          >
            Open full machine report
          </button>

          <div className="copilot-fleet-note">
            <strong>Fleet context</strong>
            <p>
              This platform is based on the project’s 10,000-observation AI4I
              monitoring dataset. Guidance is restricted to available
              statistical evidence.
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}