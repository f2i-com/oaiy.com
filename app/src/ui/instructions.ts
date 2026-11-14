/**
 * The Instructions dialog (the chat's Instructions chip): which of the
 * person's instructions this place's conversation takes, theirs from the
 * Agent's settings, its own instead, or none; and, for a kept project, whether
 * the business's tools are offered there.
 */
import type { ProjectInstructions } from '../vfs/projects';
import { INSTRUCTIONS_MAX } from '../settings';
import { projectInstructions } from '../instructions';
import { h } from './dom';
import { modal } from './modal';

export interface InstructionsChoice {
  instructions: ProjectInstructions;
  /** For a kept project: whether the business's tools are offered there (absent where the dialog did not ask). */
  businessTools?: boolean;
}

/** The person's own instructions in a line, for the dialog to say what "mine" is. */
export function instructionsPreview(text: string, max = 90): string {
  const line = text.replace(/\s+/g, ' ').trim();
  return line.length > max ? `${line.slice(0, max - 1)}…` : line;
}

export function instructionsDialog(options: {
  /** The place, as the dialog names it ("Weather report", the Front desk). */
  place: string;
  /** Its instructions as its project.json holds them (read as instructions.ts projectInstructions reads them, whatever they are). */
  current: unknown;
  /** The person's own instructions (Agent settings → Your instructions). */
  mine: string;
  /** Whether the business's tools are offered there now; null where the dialog does not ask (the Front desk, incognito). */
  businessTools: boolean | null;
  incognito: boolean;
}): Promise<InstructionsChoice | null> {
  const current = projectInstructions({ instructions: options.current });
  let use: ProjectInstructions['use'] = current.use;
  const text = h('textarea.instructions-text', {
    rows: 5,
    maxlength: INSTRUCTIONS_MAX,
    spellcheck: true,
    'aria-label': 'Instructions for this conversation only',
    placeholder: 'How the Agent should work here: what to call things, which language to answer in, what to leave alone…',
  }) as HTMLTextAreaElement;
  text.value = current.text;
  const error = h('p.modal-error', { role: 'alert' });
  text.addEventListener('input', () => (error.textContent = ''));
  const textRow = h('div.instructions-own', text);
  textRow.hidden = use !== 'these';
  const mine = instructionsPreview(typeof options.mine === 'string' ? options.mine : '');
  const radios: HTMLInputElement[] = [];
  const option = (value: ProjectInstructions['use'], label: string, hint: string) => {
    const radio = h('input', { type: 'radio', name: 'instructions-use', value, checked: value === use, onchange: () => {
      use = value;
      textRow.hidden = value !== 'these';
      error.textContent = '';
      if (value === 'these') text.focus();
    } }) as HTMLInputElement;
    radios.push(radio);
    return h('label.where-option', radio, h('span', h('strong', label), h('span.where-hint', hint)));
  };
  const business = options.businessTools === null ? null : (h('input', { type: 'checkbox', checked: options.businessTools }) as HTMLInputElement);
  const choices = h(
    'fieldset.where.instructions-use',
    h('legend', 'This conversation takes'),
    option('mine', 'Use my instructions', mine ? `From Agent settings: “${mine}”` : 'None written yet: Agent settings → Your instructions.'),
    option('these', 'Use these instead', 'Here only, in place of yours.'),
    textRow,
    option('none', 'None', 'No instructions of yours here.'),
  );
  return modal<InstructionsChoice>({
    title: 'Instructions',
    message: `What the conversation in ${options.place} is told of yours, after OAIY's own instructions (yours win where they differ on how to work or write). The phone's calls, texts and flows' tasks never get them.`,
    body: [
      choices,
      error,
      ...(business
        ? [h('label.instructions-business', business, h('span', h('strong', 'Business tools here (phone, calendar, contacts, outreach)'), h('span.where-hint', "Off: this conversation is not offered them, nor OAIY's own tools that read or change the business's calendar, contacts or phone.")))]
        : options.incognito
          ? [h('p.instructions-note', "Incognito: no business tools or business details here, and none of OAIY's own tools.")]
          : []),
    ],
    // (the text where it is chosen, else the choice that stands: a hidden text box takes no focus)
    focus: use === 'these' ? text : radios.find((r) => r.checked),
    ok: {
      label: 'Save',
      value: () => {
        if (use === 'these' && !text.value.trim()) {
          error.textContent = 'Write the instructions for this conversation, or choose another.';
          text.focus();
          return undefined;
        }
        return { instructions: { use, text: text.value }, ...(business ? { businessTools: business.checked } : {}) };
      },
    },
  });
}
