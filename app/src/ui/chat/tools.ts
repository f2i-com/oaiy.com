/**
 * Tool calls as the chat shows them: in plain words, with an icon, what they
 * worked on, and how long they took.
 */
import type { ToolCall } from '../../agent/protocol';

/** Each tool in plain words, and its icon. */
const TOOLS: Record<string, [string, string]> = {
  read_file: ['Read', 'file-text'],
  write_file: ['Wrote', 'pencil'],
  append_file: ['Added to', 'pencil'],
  edit_file: ['Edited', 'pencil'],
  delete_file: ['Deleted', 'trash'],
  list_files: ['Listed files', 'folder'],
  file_info: ['Looked at', 'info'],
  search_file: ['Searched', 'search'],
  grep: ['Searched for', 'search'],
  glob: ['Found files', 'search'],
  code_run: ['Ran code', 'terminal'],
  sandbox_shell: ['Ran', 'terminal'],
  web_fetch: ['Fetched', 'globe'],
  present_file: ['Showed', 'eye'],
  view_image: ['Looked at', 'image'],
  generate_image: ['Made a picture', 'image'],
  generate_video: ['Made a video', 'video'],
  generate_speech: ['Made speech', 'music'],
  create_voice: ['Made a voice', 'mic'],
  generate_music: ['Made music', 'music'],
  generate_sound_effect: ['Made a sound', 'music'],
  generate_3d_model: ['Made a 3D model', 'cube'],
  remove_background: ['Cut out', 'image'],
  upscale_image: ['Enlarged', 'image'],
  review_frame: ['Reviewed', 'eye'],
  media_info: ['Looked at', 'video'],
  video_frames: ['Took frames from', 'video'],
  video_split: ['Split', 'video'],
  media_compose: ['Put together', 'video'],
  softn_docs: ['Read SoftN docs', 'file-text'],
  softn_components: ['Looked up components', 'app'],
  softn_examples: ['SoftN examples', 'app'],
  softn_check: ['Checked the app', 'app'],
  softn_inspect: ['Inspected the app', 'app'],
  softn_interact: ['Used the app', 'app'],
  softn_import: ['Unpacked', 'package'],
  page_check: ['Checked the page', 'globe'],
  page_inspect: ['Inspected the page', 'globe'],
  page_interact: ['Used the page', 'globe'],
  preview_screenshot: ['Took a screenshot', 'image'],
  preview_viewport: ['Set the screen size', 'app'],
  delegate: ['Sub-agents', 'users'],
  update_plan: ['Updated the plan', 'list'],
  send_text_message: ['Texted', 'message'],
  end_call: ['Ended the call', 'phone-off'],
  request_appointment: ['Requested a booking', 'calendar'],
  lookup_business_data: ['Looked up', 'search'],
  remember: ['Remembered', 'user'],
  earlier_conversations: ['Looked back', 'clock'],
  phone_conversations: ['Phone conversations', 'phone'],
  caller_notes: ['Caller notes', 'user'],
  tell_agent: ['Told an agent', 'message'],
  start_outreach: ['Started an outreach', 'phone'],
  outreach_status: ['Outreach progress', 'list'],
  outreach_pause: ['Paused an outreach', 'clock'],
  outreach_resume: ['Resumed an outreach', 'refresh'],
  outreach_stop: ['Stopped an outreach', 'stop'],
  outreach_results: ['Outreach results', 'table'],
  record_result: ['Recorded the result', 'check'],
  calendar_free_times: ['Free times', 'calendar'],
  calendar_list: ['The calendar', 'calendar'],
  calendar_book: ['Booked', 'calendar'],
  calendar_change: ['Changed a booking', 'calendar'],
  flow_nodes: ['Flow nodes', 'flow'],
  flow_list: ['Listed flows', 'flow'],
  flow_read: ['Read a flow', 'flow'],
  flow_write: ['Wrote a flow', 'flow'],
  flow_run: ['Ran a flow', 'flow'],
  transcribe: ['Transcribed', 'mic'],
  transcribe_audio: ['Transcribed', 'mic'],
  // OAIY itself, through OAIY Desktop's control API.
  status: ["Checked OAIY's status", 'activity'],
  setup_status: ['Checked the setup', 'settings'],
  setup_finish: ['Finished the setup', 'check'],
  models_list: ['Looked at the models', 'cpu'],
  model_download_status: ['Checked the downloads', 'arrow-down'],
  model_set_default: ['Chose a model', 'cpu'],
  model_download: ['Started a download', 'arrow-down'],
  engine_start: ['Started the engine', 'power'],
  engine_stop: ['Stopped the engine', 'power'],
  engine_restart: ['Restarted the engine', 'refresh'],
  ai_sources_list: ['Looked at the AI sources', 'sparkle'],
  agent_model_set: ["Chose the Agent's model", 'sparkle'],
  chatgpt_sign_in: ['Started the ChatGPT sign-in', 'user'],
  chatgpt_sign_out: ['Signed out of ChatGPT', 'user'],
  services_list: ['Looked at the services', 'server'],
  service_logs: ["Read a service's log", 'file-text'],
  service_install: ['Installed a service', 'server'],
  service_start: ['Started a service', 'power'],
  service_stop: ['Stopped a service', 'power'],
  service_uninstall: ['Removed a service', 'trash'],
  plugins_list: ['Looked at the plugins', 'plug'],
  plugin_catalog: ['Looked at the plugin catalog', 'plug'],
  plugin_settings_get: ["Read a plugin's settings", 'settings'],
  plugin_settings_set: ["Changed a plugin's settings", 'settings'],
  plugin_setup_status: ["Checked a plugin's setup", 'list'],
  plugin_setup_open: ['Showed you a setup step', 'eye'],
  plugin_setup_step_done: ['Marked a setup step done', 'check'],
  plugin_setup_finish: ["Finished a plugin's setup", 'check'],
  plugin_install: ['Installed a plugin', 'plug'],
  plugin_enable: ['Switched a plugin on', 'power'],
  plugin_disable: ['Switched a plugin off', 'power'],
  plugin_restart: ['Restarted a plugin', 'refresh'],
  plugin_uninstall: ['Removed a plugin', 'trash'],
  plugin_command: ['Sent a plugin a command', 'send'],
  flows_list: ['Listed flows', 'flow'],
  flow_get: ['Read a flow', 'flow'],
  flow_create: ['Made a flow', 'flow'],
  flow_update: ['Changed a flow', 'flow'],
  flow_delete: ['Deleted a flow', 'trash'],
  calendar_settings_get: ['Read the calendar settings', 'calendar'],
  calendar_settings_set: ['Changed the calendar settings', 'calendar'],
  link_status: ['Checked the FormLogic link', 'link'],
  link_sync_now: ['Synced with FormLogic', 'refresh'],
  ui_open: ['Showed you a page in OAIY', 'eye'],
  logs_tail: ['Read a log', 'file-text'],
};

/** A tool's name in plain words ("Read", "Ran code"); a flow's own tool as its name reads. */
export function toolLabel(name: string): string {
  // OAIY's own tool given a prefix (another of the conversation's tools had its name) reads as itself.
  const known = TOOLS[name]?.[0] ?? (name.startsWith('oaiy_') ? TOOLS[name.slice(5)]?.[0] : undefined);
  if (known) return known;
  const words = name.replace(/[_-]+/g, ' ').trim();
  return words ? words[0].toUpperCase() + words.slice(1) : 'Tool';
}

/** A tool's icon's name. */
export function toolIcon(name: string): string {
  return TOOLS[name]?.[1] ?? (name.startsWith('oaiy_') ? TOOLS[name.slice(5)]?.[1] : undefined) ?? (name.startsWith('flow') ? 'flow' : name.startsWith('softn') ? 'app' : 'wrench');
}

/** What a call worked on, in a few words: the file, the command, the search. */
export function summarizeCall(call: ToolCall): string {
  const i = call.input;
  const s = (k: string) => (typeof i[k] === 'string' ? String(i[k]) : '');
  switch (call.name) {
    case 'read_file': case 'write_file': case 'append_file': case 'edit_file': case 'delete_file': case 'list_files': return s('path') || '/';
    case 'review_frame': return `${s('path')}${i.accept === true ? ' (accept)' : ''}`;
    case 'grep': return `${s('pattern')}${s('path') ? ` in ${s('path')}` : ''}`;
    case 'glob': return s('pattern');
    case 'sandbox_shell': return s('command').split('\n')[0];
    case 'code_run': return `${s('language') || 'javascript'}${s('file') ? ` ${s('file')}` : ''}`;
    case 'web_fetch': return s('url');
    case 'delegate': return Array.isArray(i.tasks) ? `${i.tasks.length} task${i.tasks.length === 1 ? '' : 's'}` : '';
    case 'present_file': case 'view_image': case 'file_info': case 'search_file': return s('path');
    case 'softn_import': return s('path');
    case 'softn_check': case 'softn_inspect': return s('app');
    case 'softn_interact': return Array.isArray(i.actions) ? i.actions.map((a) => Object.entries(a as Record<string, unknown>).filter(([k]) => k !== 'nth').map(([k, v]) => (k === 'value' ? `"${v}"` : `${k} ${typeof v === 'string' ? `"${v}"` : v}`)).join(' ')).join(', ') : '';
    case 'softn_docs': return s('search') ? `search: ${s('search')}` : s('topic') || s('section') || 'map';
    case 'softn_components': return Array.isArray(i.names) ? i.names.join(', ') : s('names');
    case 'softn_examples': return [s('name'), s('file'), s('install_to') && `→ ${s('install_to')}`].filter(Boolean).join(' ') || 'list';
    case 'lookup_business_data': return s('question');
    case 'request_appointment': return [s('service'), s('date'), s('time')].filter(Boolean).join(' · ');
    case 'remember': return s('name') || s('fact');
    case 'earlier_conversations': return s('words');
    case 'end_call': return i.silent === true ? 'hung up without a word' : s('goodbye');
    case 'start_outreach': return [s('kind'), Array.isArray(i.people) ? `${i.people.length} ${i.people.length === 1 ? 'person' : 'people'}` : '', s('name')].filter(Boolean).join(' · ');
    case 'outreach_status': case 'outreach_pause': case 'outreach_resume': case 'outreach_stop': return s('id');
    case 'outreach_results': return [s('id'), s('format')].filter(Boolean).join(' · ');
    case 'record_result': {
      const answers = i.answers && typeof i.answers === 'object' ? Object.entries(i.answers as Record<string, unknown>).map(([k, v]) => `${k}: ${v === true ? 'yes' : v === false ? 'no' : String(v)}`) : [];
      return [s('outcome').replace(/_/g, ' '), ...answers].filter(Boolean).join(' · ');
    }
    case 'calendar_free_times': return s('from');
    case 'generate_image': case 'generate_video': case 'generate_music': case 'generate_sound_effect': case 'generate_3d_model': return s('path') || s('output') || '';
    case 'remove_background': case 'upscale_image': case 'media_info': case 'video_frames': case 'video_split': return s('path') || s('input') || '';
    // OAIY itself: which plugin, service, model or page.
    case 'plugin_setup_open': case 'plugin_setup_step_done': return [s('pluginId'), s('stepId')].filter(Boolean).join(' · ');
    case 'plugin_setup_status': case 'plugin_settings_get': case 'plugin_setup_finish': return s('pluginId');
    case 'plugin_settings_set': return [s('pluginId'), i.settings && typeof i.settings === 'object' ? Object.keys(i.settings).join(', ') : ''].filter(Boolean).join(' · ');
    case 'plugin_command': return [s('pluginId'), s('command')].filter(Boolean).join(' · ');
    case 'plugin_enable': case 'plugin_disable': case 'plugin_restart': case 'plugin_uninstall':
    case 'service_install': case 'service_start': case 'service_stop': case 'service_uninstall': case 'service_logs':
    case 'flow_get': case 'flow_delete': return s('id');
    case 'plugin_install': return s('source').split(/[\\/]/).filter(Boolean).pop() ?? '';
    case 'models_list': return s('group');
    case 'model_set_default': return [s('group'), s('model')].filter(Boolean).join(' → ');
    case 'model_download': return s('catalogId');
    case 'agent_model_set': return [s('source') === 'chatgpt' ? 'ChatGPT' : s('source') === 'engine' ? 'the engine' : s('source'), s('model')].filter(Boolean).join(' · ');
    case 'ui_open': return s('view');
    case 'logs_tail': return s('source');
    case 'calendar_settings_set': return Object.keys(i).join(', ');
    default: return '';
  }
}

/** The prompt a picture, clip or sound is made from, shown on its card without opening it. */
export function mediaPrompt(call: ToolCall): string {
  if (!/^generate_(image|video|speech|music)$/.test(call.name)) return '';
  const i = call.input;
  const text = [i.prompt, call.name === 'generate_speech' ? i.input : null, i.lyrics].filter((v): v is string => typeof v === 'string' && !!v.trim());
  return text.join('\n\n');
}

/** How long something took, short: "340 ms", "2.4 s", "1 min 5 s". */
export function duration(ms: number): string {
  if (ms < 1000) return `${Math.max(1, Math.round(ms))} ms`;
  if (ms < 10_000) return `${(ms / 1000).toFixed(1)} s`;
  if (ms < 60_000) return `${Math.round(ms / 1000)} s`;
  const s = Math.round(ms / 1000);
  return `${Math.floor(s / 60)} min${s % 60 ? ` ${s % 60} s` : ''}`;
}

/** A tool's arguments, a line each: a string as it is (it may be a whole file), anything else as JSON. */
export function argLines(input: Record<string, unknown>): Array<{ key: string; value: string; block: boolean }> {
  return Object.entries(input).map(([key, value]) => {
    const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2) ?? String(value);
    return { key, value: text, block: text.includes('\n') || text.length > 72 };
  });
}

/** Text cut to about `lines` lines and `chars` characters, and how much was left out. */
export function clip(text: string, lines = 24, chars = 2400): { text: string; more: number } {
  let cut = text.length > chars ? text.slice(0, chars) : text;
  const rows = cut.split('\n');
  if (rows.length > lines) cut = rows.slice(0, lines).join('\n');
  return { text: cut, more: text.length - cut.length };
}

/** A tool's result as it reads best: JSON laid out, anything else as it is. */
export function prettyResult(content: string): string {
  const t = content.trim();
  if (!/^[[{]/.test(t)) return content;
  try {
    return JSON.stringify(JSON.parse(t), null, 2);
  } catch {
    return content;
  }
}
