import { useEffect, useRef, useState, type ReactNode } from 'react';
import {
  Bot,
  Download,
  FileJson,
  Globe2,
  Import,
  Maximize2,
  Minimize2,
  Save,
  Send,
  Trash2,
  Workflow,
} from 'lucide-react';
import {
  chatWithAgent,
  createTemplate,
  deleteTemplate,
  listTemplates,
} from './api';
import type {
  AgentChatResponse,
  BenchmarkScene,
  BenchmarkTemplate,
  MarsAgentModel,
} from './types';

export type MarsMode = 'studio' | 'agent' | 'templates';

interface Props {
  mode: MarsMode;
  studio: ReactNode;
  scene: BenchmarkScene | null;
  expanded: boolean;
  onExpandedChange: (expanded: boolean) => void;
  onImportScene: (scene: BenchmarkScene, source: string) => void;
}

interface ChatTurn {
  role: 'user' | 'agent';
  text: string;
  response?: AgentChatResponse;
}

export default function MarsModePanel({
  mode,
  studio,
  scene,
  expanded,
  onExpandedChange,
  onImportScene,
}: Props) {
  const [threadId, setThreadId] = useState<string>();
  const [model, setModel] = useState<MarsAgentModel>('gemini-3.1-flash-lite');
  const [webSearch, setWebSearch] = useState(false);
  const [input, setInput] = useState('');
  const [turns, setTurns] = useState<ChatTurn[]>([{
    role: 'agent',
    text: 'Describe the robots, task arrivals, and optimization goal. I will clarify critical constraints and build a workflow draft aligned with Studio.',
  }]);
  const [agentBusy, setAgentBusy] = useState(false);
  const [agentPhase, setAgentPhase] = useState<AgentChatResponse['phase']>('discovery');
  const [panelError, setPanelError] = useState<string | null>(null);
  const [templates, setTemplates] = useState<BenchmarkTemplate[]>([]);
  const [templateName, setTemplateName] = useState('');
  const [templateDescription, setTemplateDescription] = useState('');
  const [templateBusy, setTemplateBusy] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const chatEnd = useRef<HTMLDivElement>(null);

  async function refreshTemplates() {
    try {
      const result = await listTemplates();
      setTemplates(result.templates);
    } catch (reason) {
      setPanelError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  useEffect(() => {
    if (mode === 'templates') void refreshTemplates();
  }, [mode]);

  useEffect(() => {
    if (typeof chatEnd.current?.scrollIntoView === 'function') {
      chatEnd.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [turns, agentBusy]);

  async function sendMessage(
    action: 'message' | 'confirm' | 'restart' = 'message',
    explicitMessage?: string,
  ) {
    const message = explicitMessage ?? input.trim();
    if (!message || agentBusy) return;
    if (action === 'message') setInput('');
    setPanelError(null);
    setAgentBusy(true);
    if (action === 'message' && threadId) setAgentPhase('planning');
    setTurns((current) => [...current, { role: 'user', text: message }]);
    try {
      const response = await chatWithAgent({
        thread_id: threadId,
        message,
        model,
        enable_web_search: webSearch,
        current_scene: scene ?? undefined,
        action,
      });
      setThreadId(response.thread_id);
      setAgentPhase(response.phase);
      setTurns((current) => [...current, {
        role: 'agent',
        text: response.message,
        response,
      }]);
    } catch (reason) {
      setPanelError(
        reason instanceof DOMException && reason.name === 'AbortError'
          ? 'This modelling step exceeded 50 seconds. The request was cancelled; you can retry without losing earlier conversation.'
          : reason instanceof Error ? reason.message : String(reason),
      );
    } finally {
      setAgentBusy(false);
    }
  }

  async function saveCurrentScene() {
    if (!scene || templateBusy) return;
    setTemplateBusy(true);
    setPanelError(null);
    try {
      const created = await createTemplate({
        name: templateName.trim() || scene.title,
        description: templateDescription.trim() || scene.natural_language_description,
        tags: [scene.scenario_type, scene.difficulty],
        scene,
      });
      setTemplates((current) => [created, ...current]);
      setTemplateName('');
      setTemplateDescription('');
    } catch (reason) {
      setPanelError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setTemplateBusy(false);
    }
  }

  async function removeTemplate(templateId: string) {
    setPanelError(null);
    try {
      await deleteTemplate(templateId);
      setTemplates((current) => current.filter((item) => item.id !== templateId));
    } catch (reason) {
      setPanelError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  function downloadTemplate(template: BenchmarkTemplate) {
    const blob = new Blob([JSON.stringify(template, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${template.id}.benchmark.template.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  async function importTemplateFile(file?: File) {
    if (!file) return;
    setPanelError(null);
    try {
      const payload = JSON.parse(await file.text()) as BenchmarkTemplate | BenchmarkScene;
      const imported = 'scene' in payload ? payload.scene : payload;
      if (!imported || !Array.isArray(imported.tasks) || !Array.isArray(imported.nodes)) {
        throw new Error('File is not a MARS benchmark template.');
      }
      onImportScene(imported, file.name);
    } catch (reason) {
      setPanelError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  return (
    <div className="mode-panel-root">
      <div className="mode-panel-view" hidden={mode !== 'studio'}>{studio}</div>
      <div className="mode-panel-view agent-panel" hidden={mode !== 'agent'}>
        <div className="panel-toolbar">
          <div>
            <strong><Bot size={15} /> Modelling copilot</strong>
            <small>memory / structured workflow / retrieval</small>
          </div>
          <button
            type="button"
            className="icon-button"
            onClick={() => onExpandedChange(!expanded)}
            aria-label={expanded ? 'Shrink MARS Agent' : 'Expand MARS Agent'}
            title={expanded ? 'Shrink to sidebar' : 'Expand to two thirds'}
          >
            {expanded ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
          </button>
        </div>
        <div className="agent-options">
          <select value={model} onChange={(event) => setModel(event.target.value as MarsAgentModel)} aria-label="Agent model">
            <option value="gemini-3.1-flash-lite">gemini-3.1-flash-lite (recommended)</option>
            <option value="deepseek-v4-flash">deepseek-v4-flash</option>
          </select>
          <label title="When enabled, MARS sends generic workflow keywords to arXiv and includes the results in planning.">
            <input type="checkbox" checked={webSearch} onChange={(event) => setWebSearch(event.target.checked)} />
            <Globe2 size={12} /> Retrieval via arXiv
          </label>
        </div>
        <div className="agent-phase" aria-label="Agent modelling phase">
          {(['discovery', 'planning', 'review', 'ready'] as const).map((phase) => (
            <span key={phase} className={phase === agentPhase ? 'active' : ''}>{phase}</span>
          ))}
        </div>
        <div className="agent-thread" aria-live="polite">
          {turns.map((turn, index) => (
            <div className={`chat-turn ${turn.role}`} key={`${turn.role}-${index}`}>
              <span>{turn.role === 'agent' ? 'MARS' : 'You'}</span>
              <p>{turn.text}</p>
              {turn.response && (
                <AgentResult
                  response={turn.response}
                  onImport={onImportScene}
                  onConfirm={() => void sendMessage('confirm', 'Confirm and compile this atomic-task plan')}
                  canConfirm={index === turns.length - 1 && !agentBusy}
                />
              )}
            </div>
          ))}
          {agentBusy && <div className="agent-thinking"><i /><i /><i /> {agentPhase === 'discovery' ? 'Planning atomic tasks with APIYI...' : 'Processing this modelling step...'}</div>}
          <div ref={chatEnd} />
        </div>
        {panelError && <div className="panel-error" role="alert">{panelError}</div>}
        <div className="agent-composer">
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                void sendMessage();
              }
            }}
            placeholder="Example: two vending robots receive pickup tasks one minute apart..."
            aria-label="Message MARS Agent"
          />
          <button type="button" onClick={() => void sendMessage()} disabled={agentBusy || !input.trim()} aria-label="Send to MARS Agent">
            <Send size={15} />
          </button>
        </div>
      </div>

      <div className="mode-panel-view templates-panel" hidden={mode !== 'templates'}>
        <div className="panel-toolbar">
          <div>
            <strong><FileJson size={15} /> Benchmark library</strong>
            <small>versioned Studio scene snapshots</small>
          </div>
          <button type="button" className="icon-button" onClick={() => fileInput.current?.click()} aria-label="Import template file">
            <Import size={15} />
          </button>
          <input ref={fileInput} type="file" accept=".json,.template" hidden onChange={(event) => void importTemplateFile(event.target.files?.[0])} />
        </div>
        <section className="template-save-card">
          <strong>Save current Studio workflow</strong>
          <input value={templateName} onChange={(event) => setTemplateName(event.target.value)} placeholder={scene?.title ?? 'Template name'} aria-label="Template name" />
          <textarea value={templateDescription} onChange={(event) => setTemplateDescription(event.target.value)} placeholder="Benchmark purpose and expected behavior" aria-label="Template description" />
          <button type="button" onClick={() => void saveCurrentScene()} disabled={!scene || templateBusy}>
            <Save size={14} /> {templateBusy ? 'Saving...' : 'Save benchmark'}
          </button>
        </section>
        {panelError && <div className="panel-error" role="alert">{panelError}</div>}
        <div className="template-list">
          {templates.length === 0 && <div className="template-empty"><Workflow size={24} />No saved benchmarks yet.</div>}
          {templates.map((template) => (
            <article className="template-card" key={template.id}>
              <div className="template-card-title">
                <div><strong>{template.name}</strong><small>{template.scene.tasks.length} tasks / {template.scene.nodes.length} nodes</small></div>
                <span>{template.schema_version.replace('mars.benchmark.template.', '')}</span>
              </div>
              <p>{template.description || template.scene.natural_language_description}</p>
              <div className="template-tags">{template.tags.map((tag) => <span key={tag}>{tag}</span>)}</div>
              <div className="template-actions">
                <button type="button" className="primary" onClick={() => onImportScene(template.scene, template.name)}><Import size={13} /> Import to Studio</button>
                <button type="button" onClick={() => downloadTemplate(template)} aria-label={`Export ${template.name}`}><Download size={13} /></button>
                <button type="button" onClick={() => void removeTemplate(template.id)} aria-label={`Delete ${template.name}`}><Trash2 size={13} /></button>
              </div>
            </article>
          ))}
        </div>
      </div>
    </div>
  );
}

function AgentResult({
  response,
  onImport,
  onConfirm,
  canConfirm,
}: {
  response: AgentChatResponse;
  onImport: (scene: BenchmarkScene, source: string) => void;
  onConfirm: () => void;
  canConfirm: boolean;
}) {
  return (
    <div className="agent-result">
      <div className="agent-provenance">
        <small className={`provenance-badge ${response.provenance}`}>
          {response.provenance === 'api'
            ? `API / ${response.effective_model ?? response.model}`
            : response.provenance === 'api_recovered'
              ? `API recovered / ${response.effective_model ?? response.model}`
              : response.provenance === 'local_fallback'
                ? 'Local fallback / API failed'
                : 'Local intake / no API result'}
        </small>
        {response.diagnostic && <details><summary>Diagnostic</summary><p>{response.diagnostic}</p></details>}
      </div>
      <div className="agent-progress"><span style={{ width: `${response.progress}%` }} /><small>{response.phase} / {response.progress}%</small></div>
      {response.atomic_tasks.length > 0 && (
        <div className="atomic-plan">
          <strong>Atomic task plan</strong>
          {response.atomic_tasks.map((task) => (
            <div className="atomic-task" key={task.id}>
              <span>{task.id.replace('task_', '')}</span>
              <div>
                <strong>{task.name}</strong>
                <small>{task.source_robot_id} / {task.arrival_time_ms} ms / depends on {task.dependencies.join(', ') || 'none'}</small>
              </div>
            </div>
          ))}
        </div>
      )}
      {response.insights.length > 0 && <div className="agent-insights"><strong>Engineering insights</strong><ul>{response.insights.map((item) => <li key={item}>{item}</li>)}</ul></div>}
      {response.suggested_nodes.length > 0 && <div className="node-chips">{response.suggested_nodes.map((item) => <span key={item}>{item.replace(/_/g, ' ')}</span>)}</div>}
      {response.questions.length > 0 && <div className="agent-questions"><strong>Questions to refine</strong>{response.questions.map((item) => <p key={item}>{item}</p>)}</div>}
      {response.sources.some((source) => source.kind === 'web') && <details><summary>Retrieved methods</summary>{response.sources.filter((source) => source.kind === 'web').map((source) => <a href={source.url} target="_blank" rel="noreferrer" key={source.url}>{source.title}</a>)}</details>}
      {response.phase === 'review' && !response.ready_to_import && canConfirm && (
        <button type="button" className="agent-import" onClick={onConfirm}>
          <Workflow size={13} /> Confirm and compile workflow
        </button>
      )}
      {response.scene_draft && response.ready_to_import && (
        <button type="button" className="agent-import" onClick={() => onImport(response.scene_draft!, 'MARS Agent')}>
          <Import size={13} /> Import workflow to Studio
        </button>
      )}
    </div>
  );
}
