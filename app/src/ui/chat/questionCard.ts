/**
 * Interactive in-chat Question Card (`ask_user_question`) and Plan Review Card
 * (`exit_plan_mode`), styled to match OAIY's elevated card system.
 */
import type { UserQuestionAnswer, UserQuestionItem, UserQuestionResult } from '../../agent/askUser';
import type { PlanReviewDecision, PlanReviewRequest } from '../../agent/planMode';
import { h } from '../dom';
import { renderMarkdown } from '../markdown';

/**
 * Render an interactive multi-question card in the chat stream and resolve when
 * the user submits or skips (or reject/skip if aborted).
 */
export function mountQuestionCard(
  container: HTMLElement,
  questions: readonly UserQuestionItem[],
  signal?: AbortSignal,
): Promise<UserQuestionResult> {
  return new Promise((resolve) => {
    const selections = new Map<string, Set<string>>();
    const customTexts = new Map<string, string>();
    const card = h('div', { class: 'ask-card' });

    const head = h(
      'div',
      { class: 'ask-card-head' },
      h('span', { class: 'ask-card-badge' }, 'Question'),
      h('span', { class: 'ask-card-title' }, questions.length === 1 ? 'Clarification needed' : `${questions.length} questions`),
    );
    card.append(head);

    const body = h('div', { class: 'ask-card-body' });
    const inputs: HTMLInputElement[] = [];

    for (const q of questions) {
      const sel = new Set<string>();
      selections.set(q.id, sel);

      const block = h('div', { class: 'ask-q' });
      if (q.header) block.append(h('div', { class: 'ask-q-header' }, q.header));
      block.append(h('div', { class: 'ask-q-text' }, q.question));

      if (q.options?.length) {
        const optsEl = h('div', { class: 'ask-options' });
        const buttons: HTMLButtonElement[] = [];
        for (const opt of q.options) {
          const isRec = /\(recommended\)/i.test(opt.label);
          const btn = h(
            'button',
            { type: 'button', class: `ask-opt${isRec ? ' recommended' : ''}` },
            h('span', { class: 'ask-opt-label' }, opt.label),
            ...(opt.description ? [h('span', { class: 'ask-opt-desc' }, opt.description)] : []),
          ) as HTMLButtonElement;
          btn.onclick = () => {
            if (q.multiSelect) {
              if (sel.has(opt.label)) {
                sel.delete(opt.label);
                btn.classList.remove('selected');
              } else {
                sel.add(opt.label);
                btn.classList.add('selected');
              }
            } else {
              sel.clear();
              for (const b of buttons) b.classList.remove('selected');
              sel.add(opt.label);
              btn.classList.add('selected');
            }
          };
          buttons.push(btn);
          optsEl.append(btn);
        }
        block.append(optsEl);
      }

      const customInput = h('input', {
        type: 'text',
        class: 'ask-custom-input',
        placeholder: q.options?.length ? 'Or type a custom answer…' : 'Type your answer…',
      }) as HTMLInputElement;
      customInput.oninput = () => {
        customTexts.set(q.id, customInput.value);
      };
      inputs.push(customInput);
      block.append(customInput);
      body.append(block);
    }

    card.append(body);

    const submitBtn = h('button', { type: 'button', class: 'ask-btn-primary' }, 'Send answer') as HTMLButtonElement;
    const skipBtn = h('button', { type: 'button', class: 'ask-btn-ghost' }, 'Skip (let agent decide)') as HTMLButtonElement;
    const footer = h('div', { class: 'ask-card-footer' }, skipBtn, submitBtn);
    card.append(footer);

    let done = false;
    const finish = (skipped: boolean) => {
      if (done) return;
      done = true;
      signal?.removeEventListener('abort', onAbort);
      const answers: UserQuestionAnswer[] = questions.map((q) => {
        const selected = skipped ? [] : [...(selections.get(q.id) ?? [])];
        const custom = skipped ? '' : (customTexts.get(q.id) ?? '').trim();
        return {
          id: q.id,
          selected,
          ...(custom ? { custom } : {}),
          ...(skipped || (!selected.length && !custom) ? { skipped: true } : {}),
        };
      });
      // Freeze card cleanly into a settled summary pill.
      card.classList.add('settled');
      footer.replaceChildren(
        h(
          'span',
          { class: 'ask-settled-note' },
          skipped
            ? 'Skipped — agent will choose a sensible default.'
            : answers
                .map((a) => {
                  const parts = [...a.selected, ...(a.custom ? [a.custom] : [])];
                  return parts.length ? parts.join(', ') : 'Skipped';
                })
                .join(' · '),
        ),
      );
      for (const inp of inputs) inp.disabled = true;
      card.querySelectorAll<HTMLButtonElement>('.ask-opt').forEach((b) => {
        b.disabled = true;
      });
      resolve({ answers });
    };

    const onAbort = () => finish(true);
    if (signal?.aborted) {
      finish(true);
      return;
    }
    signal?.addEventListener('abort', onAbort, { once: true });

    submitBtn.onclick = () => finish(false);
    skipBtn.onclick = () => finish(true);
    for (const inp of inputs) {
      inp.onkeydown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          finish(false);
        }
      };
    }

    container.append(card);
    inputs[0]?.focus();
  });
}

/**
 * Render a Plan Review Card (`exit_plan_mode`) in chat allowing the user to
 * approve and start execution or request revisions while staying in Plan Mode.
 */
export function mountPlanReviewCard(
  container: HTMLElement,
  request: PlanReviewRequest,
  signal?: AbortSignal,
): Promise<PlanReviewDecision> {
  return new Promise((resolve) => {
    const card = h('div', { class: 'plan-review-card' });
    card.append(
      h(
        'div',
        { class: 'plan-review-head' },
        h('span', { class: 'plan-review-badge' }, 'Plan Review'),
        h('span', { class: 'plan-review-title' }, request.title),
      ),
    );
    const body = h('div', { class: 'plan-review-body md' });
    body.innerHTML = renderMarkdown(request.plan, true);
    card.append(body);

    const feedback = h('input', {
      type: 'text',
      class: 'ask-custom-input plan-review-feedback',
      placeholder: 'Optional revision feedback (leave blank to approve as written)…',
    }) as HTMLInputElement;

    const reviseBtn = h('button', { type: 'button', class: 'ask-btn-ghost' }, 'Keep planning') as HTMLButtonElement;
    const approveBtn = h('button', { type: 'button', class: 'ask-btn-primary' }, 'Approve & execute') as HTMLButtonElement;
    const footer = h('div', { class: 'plan-review-footer' }, feedback, h('div', { class: 'plan-review-actions' }, reviseBtn, approveBtn));
    card.append(footer);

    let done = false;
    const settle = (approved: boolean, note?: string) => {
      if (done) return;
      done = true;
      signal?.removeEventListener('abort', onAbort);
      card.classList.add('settled');
      const trimmed = (note ?? feedback.value).trim();
      footer.replaceChildren(
        h(
          'span',
          { class: `plan-review-verdict ${approved ? 'approved' : 'revising'}` },
          approved ? 'Approved — executing plan' : trimmed ? `Revising: ${trimmed}` : 'Kept in Plan Mode',
        ),
      );
      resolve({ approved, ...(trimmed ? { feedback: trimmed } : {}) });
    };

    const onAbort = () => settle(false, 'Cancelled');
    if (signal?.aborted) {
      settle(false, 'Cancelled');
      return;
    }
    signal?.addEventListener('abort', onAbort, { once: true });

    approveBtn.onclick = () => settle(true);
    reviseBtn.onclick = () => settle(false);
    feedback.onkeydown = (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        settle(feedback.value.trim().length === 0);
      }
    };

    container.append(card);
  });
}
