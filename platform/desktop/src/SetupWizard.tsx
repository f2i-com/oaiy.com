import { useState, type ReactNode } from "react";
import ChatGptConnector from "./ChatGptConnector";
import type { StepTarget, SetupStep, StepId } from "./setupGuide";

export default function SetupWizard({
  steps,
  onNavigate,
  actions,
}: {
  steps: SetupStep[];
  onNavigate: (target: StepTarget) => void;
  actions?: Partial<Record<StepId, ReactNode>>;
}) {
  const [index, setIndex] = useState(0);
  const [choice, setChoice] = useState<"codex" | "api" | "local">("codex");
  const step = steps[index];
  return (
    <div className="setup-wizard">
      <nav className="setup-wizard-nav" aria-label="Setup steps">
        {steps.map((s, i) => (
          <button
            type="button"
            key={s.id}
            className="btn-tiny"
            aria-current={i === index ? "step" : undefined}
            onClick={() => setIndex(i)}
          >
            {i + 1}.{" "}
            {s.id === "ai"
              ? "Your AI"
              : s.id === "runtime"
                ? "Runtime"
                : s.id === "plugins"
                  ? "Plugins"
                  : "FormLogic"}
          </button>
        ))}
      </nav>
      <h4>{step.title}</h4>
      <p className="form-hint">{step.detail}</p>
      {step.id === "runtime" && (
        <p className="form-hint">
          Check the runtime status below. If anything needs installing, use its
          action and wait for the status to update.
        </p>
      )}
      {step.id === "runtime" &&
        !step.done &&
        (actions?.runtime ?? (
          <button
            className="btn btn-primary"
            onClick={() => onNavigate("services")}
          >
            Check runtime setup
          </button>
        ))}
      {step.id === "ai" && (
        <>
          <div
            className="setup-wizard-nav"
            role="group"
            aria-label="Choose your AI connection"
          >
            {(["codex", "api", "local"] as const).map((c) => (
              <button
                type="button"
                key={c}
                className="btn"
                aria-pressed={choice === c}
                onClick={() => setChoice(c)}
              >
                {c === "codex"
                  ? "Codex / ChatGPT"
                  : c === "api"
                    ? "Provider API key"
                    : "Local model"}
              </button>
            ))}
          </div>
          {choice === "codex" ? (
            <ChatGptConnector />
          ) : (
            <div className="setup-wizard-choice">
              <p className="form-hint">
                {choice === "api"
                  ? "Open Providers, add your provider, enter its API key and model, then test the connection. Keys stay on this computer. Provider charges may apply."
                  : "Open Services to install or connect a local model server. Start it and download a model before selecting it in FormLogic."}
              </p>
              <button
                className="btn btn-primary"
                onClick={() =>
                  onNavigate(choice === "api" ? "providers" : "services")
                }
              >
                {choice === "api"
                  ? "Configure an API provider"
                  : "Set up a local model"}
              </button>
            </div>
          )}
        </>
      )}
      {step.id === "plugins" && (
        <p className="form-hint">
          Plugins are optional. Open Plugins to install Aokie when you want
          phone or device features. Start it and follow its pairing
          instructions.
        </p>
      )}
      {step.id === "plugins" && (
        <button className="btn" onClick={() => onNavigate("plugins")}>
          Browse plugins
        </button>
      )}
      {step.id === "connect" && (
        <ol className="setup-wizard-instructions">
          <li>In FormLogic, open Connect your AI and choose OAIY desktop.</li>
          <li>
            Click Connect in FormLogic, then approve the matching code in OAIY
            Connections.
          </li>
          <li>
            Return to FormLogic, select your default provider and check the
            saved setup. Keep OAIY open while using it.
          </li>
        </ol>
      )}
      <div className="setup-wizard-footer">
        <button
          className="btn"
          disabled={index === 0}
          onClick={() => setIndex((i) => i - 1)}
        >
          Back
        </button>
        <span className="form-hint" role="status">
          {step.done
            ? "Verified from current status"
            : step.optional
              ? "Optional step"
              : "Waiting for setup"}
        </span>
        {index < steps.length - 1 ? (
          <button
            className="btn btn-primary"
            onClick={() => setIndex((i) => i + 1)}
          >
            Next
          </button>
        ) : (
          <button className="btn" onClick={() => onNavigate("connections")}>
            Open Connections
          </button>
        )}
      </div>
    </div>
  );
}
