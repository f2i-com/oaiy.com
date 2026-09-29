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
};

/** A tool's name in plain words ("Read", "Ran code"); a flow's own tool as its name reads. */
export function toolLabel(name: string): string {
  const known = TOOLS[name]?.[0];
  if (known) return known;
  const words = name.replace(/[_-]+/g, ' ').trim();
  return words ? words[0].toUpperCase() + words.slice(1) : 'Tool';
}

/** A tool's icon's name. */
export function toolIcon(name: string): string {
  return TOOLS[name]?.[1] ?? (name.startsWith('flow') ? 'flow' : name.startsWith('softn') ? 'app' : 'wrench');
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
    case 'end_call': return s('goodbye');
    case 'calendar_free_times': return s('from');
    case 'generate_image': case 'generate_video': case 'generate_music': case 'generate_sound_effect': case 'generate_3d_model': return s('path') || s('output') || '';
    case 'remove_background': case 'upscale_image': case 'media_info': case 'video_frames': case 'video_split': return s('path') || s('input') || '';
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
