import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  fetchEventSimulatorRun,
  fetchEventSimulatorScenarios,
  type EventSimulatorRun,
  type EventSimulatorScenarioSummary,
  type EventSimulatorTraceStep,
} from "../api/client";
import styles from "./EventSimulator.module.css";

type JsonTab = "step" | "trace" | "artifacts";

function formatMs(ms: number | null | undefined) {
  if (ms == null || Number.isNaN(ms)) return "-";
  if (Math.abs(ms) >= 1000) return `${(ms / 1000).toFixed(3)} s`;
  return `${ms.toFixed(ms % 1 === 0 ? 0 : 1)} ms`;
}

function valueText(value: unknown) {
  if (value == null) return "-";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function eventLabel(step: EventSimulatorTraceStep) {
  const mapping = step.mapping;
  const raw = step.raw_event;
  if (!mapping || !raw) return "Timer flush";
  const label = valueText(mapping.display_name ?? raw.signal_ref);
  const unit = raw.unit ? ` ${valueText(raw.unit)}` : "";
  return `${label} = ${valueText(raw.value)}${unit}`;
}

function tierLabel(step: EventSimulatorTraceStep) {
  const classification = step.classification;
  const tier = classification?.tier;
  if (typeof tier === "number") return `Tier ${tier}`;
  return valueText(classification?.category ?? "timer");
}

function decisionLabel(decision: string) {
  return decision.replace(/_/g, " ");
}

export default function EventSimulator() {
  const navigate = useNavigate();
  const [scenarios, setScenarios] = useState<EventSimulatorScenarioSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [run, setRun] = useState<EventSimulatorRun | null>(null);
  const [stepIndex, setStepIndex] = useState(0);
  const [jsonTab, setJsonTab] = useState<JsonTab>("step");
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    fetchEventSimulatorScenarios()
      .then((items) => {
        if (!alive) return;
        setScenarios(items);
        setSelectedId((current) => current || items[0]?.id || "");
      })
      .catch(() => {
        if (alive) setError("Failed to load simulator scenarios.");
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    let alive = true;
    setLoading(true);
    setError(null);
    setPlaying(false);
    fetchEventSimulatorRun(selectedId)
      .then((data) => {
        if (!alive) return;
        setRun(data);
        setStepIndex(0);
        setJsonTab("step");
      })
      .catch(() => {
        if (alive) setError("Failed to run simulator scenario.");
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [selectedId]);

  useEffect(() => {
    if (!playing || !run) return;
    const id = window.setInterval(() => {
      setStepIndex((current) => {
        if (current >= run.trace.length - 1) {
          window.clearInterval(id);
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 900);
    return () => window.clearInterval(id);
  }, [playing, run]);

  const activeStep = run?.trace[stepIndex] ?? null;
  const activeTime = activeStep?.t_ms ?? 0;
  const visibleTrace = useMemo(
    () => (run ? run.trace.slice(0, stepIndex + 1) : []),
    [run, stepIndex],
  );
  const visibleNotifications = useMemo(
    () => (run ? run.notifications.filter((notification) => notification.emit_ms <= activeTime) : []),
    [run, activeTime],
  );
  const incident = run?.incidents[0] ?? null;
  const jsonPayload = useMemo(() => {
    if (!run) return {};
    if (jsonTab === "trace") return visibleTrace;
    if (jsonTab === "artifacts") return run.artifacts;
    return activeStep ?? {};
  }, [activeStep, jsonTab, run, visibleTrace]);

  function clampStep(next: number) {
    if (!run) return;
    setStepIndex(Math.max(0, Math.min(run.trace.length - 1, next)));
  }

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <button className={styles.backBtn} onClick={() => navigate("/")} type="button">
          {"<"} Main menu
        </button>
        <div>
          <div className={styles.kicker}>MMS / IEC 61850 Notification Lab</div>
          <h1>Event Simulator</h1>
        </div>
        {run && (
          <div className={styles.headerMeta}>
            <span>{run.scenario.station_name}</span>
            <span>{run.scenario.asset_name}</span>
            <span>{run.trace.length} trace steps</span>
          </div>
        )}
      </header>

      {error && <div className={styles.error}>{error}</div>}

      <main className={styles.shell}>
        <aside className={styles.scenarioPanel}>
          <div className={styles.panelTitle}>Scenarios</div>
          <div className={styles.scenarioList}>
            {scenarios.map((scenario) => (
              <button
                key={scenario.id}
                className={`${styles.scenarioBtn} ${scenario.id === selectedId ? styles.scenarioBtnActive : ""}`}
                onClick={() => setSelectedId(scenario.id)}
                type="button"
              >
                <span>{scenario.title}</span>
                <small>{scenario.subtitle}</small>
                <em>{scenario.event_count} events</em>
              </button>
            ))}
          </div>
        </aside>

        <section className={styles.workbench}>
          {loading && <div className={styles.loading}>Loading simulator...</div>}

          {!loading && run && activeStep && (
            <>
              <section className={styles.controlBand}>
                <div>
                  <div className={styles.scenarioTitle}>{run.scenario.title}</div>
                  <p>{run.scenario.description}</p>
                </div>
                <div className={styles.controls}>
                  <button onClick={() => clampStep(stepIndex - 1)} disabled={stepIndex === 0} type="button">
                    Prev
                  </button>
                  <button onClick={() => setPlaying((value) => !value)} type="button">
                    {playing ? "Pause" : "Play"}
                  </button>
                  <button onClick={() => clampStep(stepIndex + 1)} disabled={stepIndex >= run.trace.length - 1} type="button">
                    Next
                  </button>
                  <input
                    aria-label="Trace step"
                    max={run.trace.length - 1}
                    min={0}
                    onChange={(event) => clampStep(Number(event.target.value))}
                    type="range"
                    value={stepIndex}
                  />
                  <span>{stepIndex + 1}/{run.trace.length}</span>
                </div>
              </section>

              <div className={styles.summaryGrid}>
                <section className={styles.livePanel}>
                  <div className={styles.panelTitle}>Current Step</div>
                  <div className={styles.currentStep}>
                    <span className={styles.timeBadge}>{formatMs(activeStep.t_ms)}</span>
                    <h2>{eventLabel(activeStep)}</h2>
                    <dl>
                      <div>
                        <dt>Decision</dt>
                        <dd>{decisionLabel(activeStep.decision)}</dd>
                      </div>
                      <div>
                        <dt>Class</dt>
                        <dd>{tierLabel(activeStep)}</dd>
                      </div>
                      <div>
                        <dt>Reason</dt>
                        <dd>{valueText(activeStep.classification?.reason ?? "Pending timer flush")}</dd>
                      </div>
                    </dl>
                  </div>
                </section>

                <section className={styles.livePanel}>
                  <div className={styles.panelTitle}>Grouped Incident</div>
                  {incident ? (
                    <div className={styles.incidentBox}>
                      <span className={styles.tierPill}>Tier {incident.primary_tier ?? "-"}</span>
                      <h2>{incident.title}</h2>
                      <p>{incident.summary}</p>
                      <div className={styles.metrics}>
                        <span>{incident.event_count} events</span>
                        <span>{incident.measurements.length} context</span>
                        <span>{incident.artifacts.length} artifacts</span>
                      </div>
                    </div>
                  ) : (
                    <div className={styles.empty}>No incident yet.</div>
                  )}
                </section>

                <section className={styles.livePanel}>
                  <div className={styles.panelTitle}>Notification Outbox</div>
                  <div className={styles.notifList}>
                    {visibleNotifications.length === 0 && <div className={styles.empty}>No emitted notification.</div>}
                    {visibleNotifications.map((notification) => (
                      <div
                        key={notification.id}
                        className={styles.notificationItem}
                        style={{ borderLeftColor: notification.color }}
                      >
                        <span>{notification.cluster} / Tier {notification.tier}</span>
                        <strong>{notification.title}</strong>
                        <small>{formatMs(notification.emit_ms)} - {notification.timing_note}</small>
                      </div>
                    ))}
                  </div>
                </section>
              </div>

              <section className={styles.timelinePanel}>
                <div className={styles.panelTitle}>Trace Timeline</div>
                <div className={styles.timeline}>
                  {run.trace.map((step, index) => (
                    <button
                      key={`${step.step}-${step.t_ms}`}
                      className={`${styles.traceItem} ${index === stepIndex ? styles.traceItemActive : ""} ${index <= stepIndex ? styles.traceItemSeen : ""}`}
                      onClick={() => clampStep(index)}
                      type="button"
                    >
                      <span>{formatMs(step.t_ms)}</span>
                      <strong>{eventLabel(step)}</strong>
                      <small>{decisionLabel(step.decision)}</small>
                    </button>
                  ))}
                </div>
              </section>

              <section className={styles.jsonPanel}>
                <div className={styles.jsonTabs}>
                  <button className={jsonTab === "step" ? styles.jsonTabActive : ""} onClick={() => setJsonTab("step")} type="button">
                    Active Step JSON
                  </button>
                  <button className={jsonTab === "trace" ? styles.jsonTabActive : ""} onClick={() => setJsonTab("trace")} type="button">
                    Visible Trace JSON
                  </button>
                  <button className={jsonTab === "artifacts" ? styles.jsonTabActive : ""} onClick={() => setJsonTab("artifacts")} type="button">
                    Artifacts JSON
                  </button>
                </div>
                <pre>{JSON.stringify(jsonPayload, null, 2)}</pre>
              </section>
            </>
          )}
        </section>
      </main>
    </div>
  );
}
