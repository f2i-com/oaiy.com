import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from 'react';
import { BookUser, Download, FileUp, Phone, Search, StickyNote, Trash2, Undo2, Upload, UserPlus, X } from 'lucide-react';
import { contacts as api, isNotFound, type Contact, type ContactChange, type ImportReport, type ImportRow } from './api';
import {
  IMPORT_COUNTRIES,
  contactLabel,
  csvFileName,
  factId,
  initials,
  matchesContact,
  numberKey,
  numberProblem,
  readableNumber,
  saidWhen,
  skipReason,
} from './contactsModel';
import { useToast } from './Toasts';
import { useVisiblePoll } from './useVisiblePoll';

/**
 * Contacts: the people who ring and text. A searchable list, and beside it
 * one person: their name (one word is fine, and a name set here is the
 * person's, never changed by the receptionist), their number, the person's
 * notes for the receptionist (read on every call and text with them), and
 * what the receptionist remembered, each of which can be forgotten. Changes
 * wait in an unsaved-changes bar, as on Hours & Services. People are added
 * by hand, or from a CSV file (a preview first); all of them go out as one.
 */

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** What the panel beside the list shows. */
type Side = { kind: 'contact'; key: string } | { kind: 'new'; number?: string } | { kind: 'import' };

export default function ContactsPanel({ open = null, onOpened }: { open?: string | null; onOpened?: () => void }) {
  const toast = useToast();
  const [list, setList] = useState<Contact[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  /** This desktop keeps no contacts (it is older than them). */
  const [missing, setMissing] = useState(false);
  const [q, setQ] = useState('');
  const [side, setSide] = useState<Side | null>(null);
  /** The open contact has changes not saved: another is not opened over them. */
  const [dirty, setDirty] = useState(false);
  const [nudge, setNudge] = useState(0);
  const [exporting, setExporting] = useState(false);

  const load = useCallback(async () => {
    try {
      const r = await api.list();
      setList(r.contacts ?? []);
      setError(null);
      setMissing(false);
    } catch (e) {
      if (isNotFound(e)) setMissing(true);
      else setError(errText(e));
    }
  }, []);
  // Asked again now and then: calls and texts bring people in, and the receptionist remembers things.
  useVisiblePoll(() => void load(), 15_000);

  /** Show `next` beside the list, unless the open contact has changes: then its save bar says so. */
  const go = useCallback(
    (next: Side | null) => {
      const same = next?.kind === 'contact' && side?.kind === 'contact' && next.key === side.key;
      if (dirty && !same) {
        setNudge((n) => n + 1);
        return;
      }
      setSide(next);
    },
    [dirty, side],
  );

  // The Agent asked for one person (ui_open): theirs once the list has them, else a new one with their number.
  useEffect(() => {
    if (!open || !list) return;
    const found = list.find((c) => c.key === open);
    go(found ? { kind: 'contact', key: found.key } : { kind: 'new', number: readableNumber('', open) });
    onOpened?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, list === null]);

  // Escape closes the panel (not over unsaved changes).
  useEffect(() => {
    if (!side) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !dirty) setSide(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [side, dirty]);

  // A panel opened is brought into view, and its heading takes the focus.
  const sideKey = side ? (side.kind === 'contact' ? `contact:${side.key}` : side.kind) : null;
  useEffect(() => {
    if (!sideKey) return;
    const id = requestAnimationFrame(() => {
      const heading = document.getElementById('contacts-side-title');
      heading?.closest('aside')?.scrollIntoView?.({ block: 'nearest', behavior: 'smooth' });
      heading?.focus({ preventScroll: true });
    });
    return () => cancelAnimationFrame(id);
  }, [sideKey]);

  const shown = useMemo(() => (list ?? []).filter((c) => matchesContact(c, q)), [list, q]);
  const current = side?.kind === 'contact' ? list?.find((c) => c.key === side.key) ?? null : null;
  // The open contact was deleted elsewhere: nothing to show.
  useEffect(() => {
    if (side?.kind === 'contact' && list && !current) setSide(null);
  }, [side, list, current]);

  const replace = (c: Contact) => setList((l) => {
    const rest = (l ?? []).filter((x) => x.key !== c.key);
    return [...rest, c].sort(byName);
  });

  const exportCsv = async () => {
    setExporting(true);
    try {
      const blob = await api.exportCsv();
      const file = csvFileName();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = file;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
      const n = list?.length ?? 0;
      toast.push({ kind: 'success', title: 'Exported', body: `${n} ${n === 1 ? 'contact' : 'contacts'}, saved as ${file}.` });
    } catch (e) {
      toast.push({ kind: 'error', title: 'Not exported', body: errText(e) });
    } finally {
      setExporting(false);
    }
  };

  const count = list ? (q.trim() ? `${shown.length} of ${list.length}` : `${list.length} ${list.length === 1 ? 'contact' : 'contacts'}`) : '';

  return (
    <div className={`panel contacts-page${side ? ' has-side' : ''}`}>
      {error && <div className="banner banner-err">The contacts could not be read: {error}</div>}
      {missing ? (
        <div className="banner banner-pending">
          This OAIY does not keep contacts yet. Update OAIY to name the people who ring, keep notes for the receptionist, and see what it remembered.
        </div>
      ) : (
        <div className="contacts-toolbar">
          <label className="contacts-search">
            <Search size={15} aria-hidden />
            <input type="search" value={q} placeholder="Search by name, number or notes" aria-label="Search contacts" onChange={(e) => setQ(e.target.value)} />
          </label>
          <span className="contacts-count" aria-live="polite">
            {count}
          </span>
          <div className="contacts-actions">
            <button type="button" className="btn btn-ghost" onClick={() => go({ kind: 'import' })}>
              <Upload size={14} /> Import CSV
            </button>
            <button type="button" className="btn btn-ghost" disabled={!list?.length || exporting} onClick={() => void exportCsv()}>
              <Download size={14} /> {exporting ? 'Exporting…' : 'Export CSV'}
            </button>
            <button type="button" className="btn btn-primary" onClick={() => go({ kind: 'new' })}>
              <UserPlus size={14} /> Add contact
            </button>
          </div>
        </div>
      )}

      {!missing && (
        <div className={`contacts-main${side ? ' has-side' : ''}`}>
          <div className="contacts-board">
            {list === null ? (
              !error && <div className="empty-state">Reading the contacts…</div>
            ) : list.length === 0 ? (
              <div className="empty-state contacts-empty">
                <BookUser size={22} aria-hidden />
                <p>Contacts appear as people ring and text, or add one.</p>
                <p>You can name them, and leave notes the receptionist reads on every call.</p>
                <div className="contacts-empty-actions">
                  <button type="button" className="btn btn-primary" onClick={() => go({ kind: 'new' })}>
                    <UserPlus size={14} /> Add contact
                  </button>
                  <button type="button" className="btn btn-ghost" onClick={() => go({ kind: 'import' })}>
                    <Upload size={14} /> Import CSV
                  </button>
                </div>
              </div>
            ) : shown.length === 0 ? (
              <div className="empty-state empty-state-sm">
                <p>No contact matches “{q.trim()}”.</p>
              </div>
            ) : (
              <ul className="contact-list" aria-label="Contacts">
                {shown.map((c) => {
                  const on = side?.kind === 'contact' && side.key === c.key;
                  return (
                    <li key={c.key}>
                      <button type="button" className={`contact-row${on ? ' is-selected' : ''}`} aria-current={on ? 'true' : undefined} onClick={() => go({ kind: 'contact', key: c.key })}>
                        <span className={`contact-avatar${c.name ? '' : ' is-unnamed'}`} aria-hidden>
                          {initials(c.name) || <Phone size={14} />}
                        </span>
                        <span className="contact-text">
                          <strong className={c.name ? undefined : 'is-unnamed'}>{c.name || 'No name yet'}</strong>
                          <small>{readableNumber(c.number, c.key)}</small>
                        </span>
                        <span className="contact-meta">
                          {c.notes && (
                            <em title="Has notes for the receptionist">
                              <StickyNote size={11} aria-hidden /> Notes
                            </em>
                          )}
                          {c.facts.length > 0 && <em title="What the receptionist remembered">{c.facts.length} remembered</em>}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>

          {side && (
            <aside className="cal-side contacts-side" aria-labelledby="contacts-side-title">
              {side.kind === 'contact' && current && (
                <ContactEditor
                  key={current.key}
                  contact={current}
                  nudge={nudge}
                  onDirty={setDirty}
                  onSaved={replace}
                  onDeleted={(c) => {
                    setDirty(false);
                    setSide(null);
                    setList((l) => (l ?? []).filter((x) => x.key !== c.key));
                    toast.push({ kind: 'success', title: 'Deleted', body: `${contactLabel(c)} is no longer a contact.` });
                  }}
                  onClose={() => go(null)}
                />
              )}
              {side.kind === 'new' && (
                <NewContact
                  key={side.number ?? ''}
                  initialNumber={side.number}
                  existing={list ?? []}
                  onAdded={(c) => {
                    replace(c);
                    setSide({ kind: 'contact', key: c.key });
                    toast.push({ kind: 'success', title: 'Added', body: `${contactLabel(c)} is a contact now.` });
                  }}
                  onOpen={(key) => setSide({ kind: 'contact', key })}
                  onClose={() => setSide(null)}
                />
              )}
              {side.kind === 'import' && (
                <ImportPanel
                  onImported={() => {
                    setSide(null);
                    void load();
                  }}
                  onClose={() => setSide(null)}
                />
              )}
            </aside>
          )}
        </div>
      )}
    </div>
  );
}

const byName = (a: Contact, b: Contact) => {
  const [an, bn] = [a.name.toLowerCase(), b.name.toLowerCase()];
  if (!an !== !bn) return an ? -1 : 1;
  return an < bn ? -1 : an > bn ? 1 : a.key < b.key ? -1 : a.key > b.key ? 1 : 0;
};

/** The panel's heading: a kicker, a title that takes the focus, and a close button. */
function SideHead({ kicker, title, onClose }: { kicker: string; title: string; onClose: () => void }) {
  return (
    <div className="cal-side-head">
      <div className="cal-side-heading">
        <span className="cal-side-kicker">{kicker}</span>
        <h2 id="contacts-side-title" tabIndex={-1}>
          {title}
        </h2>
      </div>
      <button type="button" className="icon-button" aria-label="Close the panel" title="Close (Esc)" onClick={onClose}>
        <X size={16} />
      </button>
    </div>
  );
}

// ---- one person -------------------------------------------------------------------------------

/** A name as the desktop keeps it: its words, single-spaced. */
const keptName = (s: string) => s.trim().split(/\s+/).filter(Boolean).join(' ');
/** Notes as the desktop keeps them: one kind of line break, trimmed. */
const keptNotes = (s: string) => s.replace(/\r\n?/g, '\n').trim();
const MAX_NAME = 80;
const MAX_NOTES = 2000;

function ContactEditor({
  contact,
  nudge,
  onDirty,
  onSaved,
  onDeleted,
  onClose,
}: {
  contact: Contact;
  /** Counts up each time the page is asked to leave this contact with changes. */
  nudge: number;
  onDirty: (dirty: boolean) => void;
  onSaved: (c: Contact) => void;
  onDeleted: (c: Contact) => void;
  onClose: () => void;
}) {
  const toast = useToast();
  const [base, setBase] = useState(contact);
  const [name, setName] = useState(contact.name);
  const [notes, setNotes] = useState(contact.notes);
  /** Keep the receptionist's name as the person's own. */
  const [pin, setPin] = useState(false);
  /** The facts to forget when saved. */
  const [removed, setRemoved] = useState<Set<string>>(() => new Set());
  const [saving, setSaving] = useState(false);
  const [justSaved, setJustSaved] = useState(false);
  const [nudged, setNudged] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const saveButton = useRef<HTMLButtonElement | null>(null);

  const nameChanged = keptName(name) !== base.name;
  const notesChanged = keptNotes(notes) !== base.notes;
  const dirty = nameChanged || notesChanged || pin || removed.size > 0;
  useEffect(() => onDirty(dirty), [dirty, onDirty]);
  useEffect(() => () => onDirty(false), [onDirty]);

  // Newer from the desktop (the receptionist remembered something, the Agent named them): taken while nothing here changed.
  const incoming = JSON.stringify(contact);
  useEffect(() => {
    if (incoming === JSON.stringify(base)) return;
    setBase(contact);
    if (!dirty) {
      setName(contact.name);
      setNotes(contact.notes);
    }
    setRemoved((r) => new Set([...r].filter((id) => contact.facts.some((f) => factId(f) === id))));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [incoming]);

  // Asked to leave with changes: said in the save bar, and its Save takes the focus.
  useEffect(() => {
    if (!nudge || !dirty) return;
    setNudged(true);
    requestAnimationFrame(() => saveButton.current?.focus());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nudge]);
  useEffect(() => {
    if (!justSaved) return;
    const id = window.setTimeout(() => setJustSaved(false), 3000);
    return () => window.clearTimeout(id);
  }, [justSaved]);

  const discard = () => {
    setName(base.name);
    setNotes(base.notes);
    setPin(false);
    setRemoved(new Set());
    setNudged(false);
  };

  const save = async () => {
    setSaving(true);
    try {
      const change: ContactChange = {};
      if (nameChanged || pin) change.name = keptName(name);
      if (notesChanged) change.notes = keptNotes(notes);
      let fresh = base;
      if (Object.keys(change).length) fresh = await api.save(base.key, change);
      // The facts to forget, by where each is now (the last first), each checked by what it says.
      const doomed = fresh.facts
        .map((f, i) => ({ f, i }))
        .filter(({ f }) => removed.has(factId(f)))
        .sort((a, b) => b.i - a.i);
      for (const { f, i } of doomed) fresh = (await api.forgetFact(base.key, i, f.text)).contact;
      setBase(fresh);
      setName(fresh.name);
      setNotes(fresh.notes);
      setPin(false);
      setRemoved(new Set());
      setNudged(false);
      setJustSaved(true);
      onSaved(fresh);
    } catch (e) {
      toast.push({ kind: 'error', title: 'Not saved', body: errText(e) });
    } finally {
      setSaving(false);
    }
  };

  const remove = async () => {
    setDeleting(true);
    try {
      await api.remove(base.key);
      onDeleted(base);
    } catch (e) {
      toast.push({ kind: 'error', title: 'Not deleted', body: errText(e) });
      setDeleting(false);
    }
  };

  const toggle = (id: string, on: boolean) =>
    setRemoved((r) => {
      const next = new Set(r);
      if (on) next.add(id);
      else next.delete(id);
      return next;
    });

  const label = contactLabel(base);
  const named = keptName(name);
  const nameHint: ReactNode =
    nameChanged && named ? (
      'Saved as your name for them: the receptionist never changes it.'
    ) : nameChanged ? (
      'Cleared when you save: the receptionist can learn their name again.'
    ) : base.nameBy === 'owner' && base.name ? (
      'You named them: the receptionist never changes it.'
    ) : base.nameBy === 'agent' && base.name ? (
      <>
        The receptionist learned this name on a call.{' '}
        {pin ? (
          'It is yours once saved.'
        ) : (
          <button type="button" className="btn-link" onClick={() => setPin(true)}>
            Keep this name
          </button>
        )}
      </>
    ) : (
      'One name is fine, as they say it: “Lance”.'
    );
  const notesLeft = MAX_NOTES - notes.length;

  return (
    <>
      <SideHead kicker="Contact" title={label} onClose={onClose} />
      <div className="cal-side-body contact-editor">
        <label className="form-row contact-name">
          <span>Name</span>
          <input value={name} maxLength={MAX_NAME} placeholder="Their name, as the receptionist says it" autoComplete="off" onChange={(e) => setName(e.target.value)} />
        </label>
        <p className="form-hint contact-name-hint">{nameHint}</p>

        <div className="form-row contact-number">
          <span>Number</span>
          <p>
            <Phone size={14} aria-hidden /> <span>{readableNumber(base.number, base.key)}</span>
          </p>
        </div>

        <label className="form-row contact-notes">
          <span>Notes for the receptionist</span>
          <textarea value={notes} maxLength={MAX_NOTES} rows={5} placeholder="What it should know: how they like to be spoken to, what they usually book, anything to watch for." onChange={(e) => setNotes(e.target.value)} />
        </label>
        <p className="form-hint contact-notes-hint">
          <span>The receptionist reads these on every call and text with them.</span>
          <span className={`contact-left${notesLeft < 100 ? ' is-low' : ''}`}>{notesLeft < 400 ? `${notesLeft} left` : ''}</span>
        </p>

        <section className="contact-facts" aria-labelledby="contact-facts-title">
          <h3 className="section-title" id="contact-facts-title">
            What the receptionist remembered <span className="section-count">{base.facts.length}</span>
          </h3>
          {base.facts.length === 0 ? (
            <p className="form-hint">Nothing yet. What it learns on calls and texts with them shows here.</p>
          ) : (
            <ul className="fact-list">
              {base.facts.map((f) => {
                const id = factId(f);
                const gone = removed.has(id);
                return (
                  <li key={id} className={gone ? 'is-removed' : undefined}>
                    <span className="fact-text">
                      <span className={gone ? 'is-struck' : undefined}>{f.text}</span>
                      <small>
                        {saidWhen(f.at)}
                        {f.by === 'owner' ? ' · by you' : ''}
                        {gone ? ' · forgotten when you save' : ''}
                      </small>
                    </span>
                    {gone ? (
                      <button type="button" className="btn-tiny" onClick={() => toggle(id, false)}>
                        <Undo2 size={13} /> Keep
                      </button>
                    ) : (
                      <button type="button" className="icon-button fact-remove" aria-label={`Forget: ${f.text}`} title="Forget this" onClick={() => toggle(id, true)}>
                        <X size={14} />
                      </button>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        <div className="contact-danger">
          {confirming ? (
            <div className="contact-confirm" role="group" aria-label="Delete this contact">
              <p>
                Delete <strong>{label}</strong>? Their name, your notes and what the receptionist remembered all go. If they ring again, they come back as a new contact.
              </p>
              <div className="cal-side-actions">
                <button type="button" className="btn btn-danger" disabled={deleting} onClick={() => void remove()}>
                  <Trash2 size={14} /> {deleting ? 'Deleting…' : 'Delete'}
                </button>
                <button type="button" className="btn btn-ghost" disabled={deleting} onClick={() => setConfirming(false)}>
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <button type="button" className="btn-link contact-delete" onClick={() => setConfirming(true)}>
              <Trash2 size={13} /> Delete contact
            </button>
          )}
        </div>

        {(dirty || justSaved) && (
          <div className={`hours-savebar contacts-savebar${dirty ? ' is-dirty' : ''}${dirty && nudged ? ' is-nudged' : ''}`} role="region" aria-label="Save">
            <div className="hours-savebar-text" aria-live="polite">
              {dirty ? (
                <>
                  <strong>
                    <i aria-hidden /> Unsaved changes
                  </strong>
                  <small>{nudged ? `Save or discard the changes to ${label} first.` : 'The receptionist keeps the saved ones until you save.'}</small>
                </>
              ) : (
                <strong className="setup-ok">Saved</strong>
              )}
            </div>
            {dirty && (
              <>
                <button type="button" className="btn btn-ghost" disabled={saving} onClick={discard}>
                  Discard
                </button>
                <button ref={saveButton} type="button" className="btn btn-primary" disabled={saving} onClick={() => void save()}>
                  {saving ? 'Saving…' : 'Save changes'}
                </button>
              </>
            )}
          </div>
        )}
      </div>
    </>
  );
}

// ---- a new person ------------------------------------------------------------------------------

function NewContact({
  initialNumber,
  existing,
  onAdded,
  onOpen,
  onClose,
}: {
  initialNumber?: string;
  existing: Contact[];
  onAdded: (c: Contact) => void;
  onOpen: (key: string) => void;
  onClose: () => void;
}) {
  const toast = useToast();
  const [name, setName] = useState('');
  const [number, setNumber] = useState(initialNumber ?? '');
  const [tried, setTried] = useState(false);
  const [saving, setSaving] = useState(false);
  const problem = numberProblem(number);
  const key = numberKey(number);
  const already = !problem && key ? existing.find((c) => c.key === key) : undefined;

  const add = async (e: FormEvent) => {
    e.preventDefault();
    setTried(true);
    if (problem || already) return;
    setSaving(true);
    try {
      const change: ContactChange = { number: number.trim() };
      if (keptName(name)) change.name = keptName(name);
      onAdded(await api.save(number.trim(), change));
    } catch (err) {
      toast.push({ kind: 'error', title: 'Not added', body: errText(err) });
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      <SideHead kicker="New contact" title="Add a contact" onClose={onClose} />
      <div className="cal-side-body">
        <form className="contact-form" onSubmit={(e) => void add(e)} noValidate>
          <label className="form-row contact-name">
            <span>Name</span>
            <input value={name} maxLength={MAX_NAME} placeholder="Lance" autoComplete="off" onChange={(e) => setName(e.target.value)} />
          </label>
          <p className="form-hint">One name is fine. The receptionist greets them by it, and never changes it.</p>
          <label className="form-row">
            <span>Number</span>
            <input
              value={number}
              inputMode="tel"
              placeholder="0491 570 006"
              autoComplete="off"
              aria-invalid={tried && !!problem}
              onChange={(e) => setNumber(e.target.value)}
            />
          </label>
          {tried && problem ? (
            <small className="hours-error" role="alert">
              {problem}
            </small>
          ) : (
            <p className="form-hint">Any way you write it: 0491 570 006, or +61 491 570 006.</p>
          )}
          {already && (
            <div className="contact-already" role="status">
              <span>
                Already a contact: <strong>{contactLabel(already)}</strong>.
              </span>
              <button type="button" className="btn-link" onClick={() => onOpen(already.key)}>
                Open it
              </button>
            </div>
          )}
          <div className="cal-side-actions">
            <button type="submit" className="btn btn-primary" disabled={saving || !!already}>
              <UserPlus size={14} /> {saving ? 'Adding…' : 'Add contact'}
            </button>
            <button type="button" className="btn btn-ghost" onClick={onClose}>
              Cancel
            </button>
          </div>
        </form>
      </div>
    </>
  );
}

// ---- a CSV file -------------------------------------------------------------------------------

/** The largest file read (the desktop takes 8 MB). */
const MAX_FILE = 8 * 1024 * 1024;

/** A file's text: UTF-8, else (a spreadsheet's own export) Windows-1252. */
async function fileText(f: File): Promise<string> {
  const bytes = typeof f.arrayBuffer === 'function' ? new Uint8Array(await f.arrayBuffer()) : null;
  if (!bytes) return f.text();
  const utf8 = new TextDecoder('utf-8').decode(bytes);
  if (!utf8.includes('�')) return utf8;
  try {
    return new TextDecoder('windows-1252').decode(bytes);
  } catch {
    return utf8;
  }
}

const ACTION: Record<ImportRow['action'], string> = { add: 'New', update: 'Update', unchanged: 'Same', skip: 'Skip' };

function ImportPanel({ onImported, onClose }: { onImported: (r: ImportReport) => void; onClose: () => void }) {
  const toast = useToast();
  const [file, setFile] = useState<{ name: string; text: string } | null>(null);
  const [country, setCountry] = useState('AU');
  const [replaceNames, setReplaceNames] = useState(false);
  const [report, setReport] = useState<ImportReport | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [busy, setBusy] = useState<'reading' | 'importing' | null>(null);
  const seq = useRef(0);

  // The file as it would go, again whenever the file, the country or "replace" changes. Nothing is written.
  useEffect(() => {
    if (!file) return;
    const n = ++seq.current;
    setBusy('reading');
    setProblem(null);
    api
      .importCsv({ csv: file.text, country, replaceNames, preview: true })
      .then(
        (r) => n === seq.current && setReport(r),
        (e) => {
          if (n !== seq.current) return;
          setReport(null);
          setProblem(isNotFound(e) ? 'This OAIY cannot import contacts yet: update OAIY first.' : errText(e));
        },
      )
      .finally(() => n === seq.current && setBusy(null));
  }, [file, country, replaceNames]);

  const choose = async (f: File | undefined) => {
    if (!f) return;
    setReport(null);
    if (f.size > MAX_FILE) {
      setFile(null);
      setProblem('That file is over 8 MB: split it, and import each part.');
      return;
    }
    setFile({ name: f.name, text: await fileText(f) });
  };

  const changes = report ? report.added + report.updated : 0;
  const run = async () => {
    if (!file || !report || !changes) return;
    setBusy('importing');
    try {
      const r = await api.importCsv({ csv: file.text, country, replaceNames, preview: false });
      const parts = [`${r.added} added`, `${r.updated} updated`, r.unchanged ? `${r.unchanged} already here` : '', r.skipped ? `${r.skipped} skipped` : ''].filter(Boolean);
      toast.push({ kind: 'success', title: 'Imported', body: `${parts.join(', ')}.` });
      onImported(r);
    } catch (e) {
      toast.push({ kind: 'error', title: 'Not imported', body: errText(e) });
      setBusy(null);
    }
  };

  return (
    <>
      <SideHead kicker="Import" title="Contacts from a CSV file" onClose={onClose} />
      <div className="cal-side-body contact-import">
        <label className="form-row import-file">
          <span>A CSV file</span>
          <input type="file" accept=".csv,text/csv,text/plain" onChange={(e) => void choose(e.target.files?.[0])} />
        </label>
        <label className="form-row">
          <span>Numbers without a country code are from</span>
          <select value={country} onChange={(e) => setCountry(e.target.value)}>
            {IMPORT_COUNTRIES.map((c) => (
              <option key={c.code} value={c.code}>
                {c.name}
              </option>
            ))}
          </select>
        </label>
        {!file && (
          <p className="form-hint">
            A header row names the columns: name (or first and last name), number (or phone, mobile) and notes. A Google Contacts export works as it is, and so does a file exported here.
          </p>
        )}
        {problem && (
          <div className="banner banner-err" role="alert">
            {problem}
          </div>
        )}
        {file && !report && !problem && <p className="form-hint">Reading {file.name}…</p>}

        {report && (
          <>
            <div className="import-stats" aria-label="What the import does">
              <ImportStat n={report.added} label="new" tone="add" />
              <ImportStat n={report.updated} label="to update" tone="update" />
              <ImportStat n={report.unchanged} label="already here" tone="same" />
              <ImportStat n={report.skipped} label="skipped" tone="skip" />
            </div>
            {report.skipped > 0 && (
              <ul className="import-reasons" aria-label="Why rows are skipped">
                {Object.entries(report.reasons).map(([reason, n]) => (
                  <li key={reason}>{skipReason(reason, n)}</li>
                ))}
              </ul>
            )}
            <label className="contacts-switch">
              <input type="checkbox" role="switch" className="switch" checked={replaceNames} onChange={(e) => setReplaceNames(e.target.checked)} />
              <span>Replace names I’ve set</span>
            </label>
            <p className="form-hint">
              {replaceNames ? 'The file’s names replace yours.' : 'A name you set stays; the file names the rest.'} Notes in the file are added after the notes there.
            </p>
            <h3 className="section-title">
              The first rows <span className="section-count">{report.rows}</span>
            </h3>
            <ol className="import-rows">
              {report.sample.map((row) => (
                <li key={row.row} className={`is-${row.action}`}>
                  <span className="import-row-no">{row.row}</span>
                  <span className="import-row-text">
                    <strong>{row.name || 'No name'}</strong>
                    <small>
                      {row.number ? readableNumber(row.number) : 'no number'}
                      {row.keptName ? ` · keeps “${row.keptName}”` : ''}
                      {row.why ? ` · ${row.why}` : ''}
                    </small>
                  </span>
                  <em className="import-tag">{ACTION[row.action]}</em>
                </li>
              ))}
            </ol>
            {report.columns.length > 0 && (
              <p className="form-hint">
                Read from {report.columns.join(', ')}
                {report.headerRow ? `, the header on line ${report.headerRow}` : ''}.
              </p>
            )}
            <div className="cal-side-actions import-actions">
              <button type="button" className="btn btn-primary" disabled={busy !== null || !changes} onClick={() => void run()}>
                <FileUp size={14} /> {busy === 'importing' ? 'Importing…' : changes ? `Import ${changes} ${changes === 1 ? 'contact' : 'contacts'}` : 'Nothing to import'}
              </button>
              <button type="button" className="btn btn-ghost" disabled={busy === 'importing'} onClick={onClose}>
                Cancel
              </button>
            </div>
          </>
        )}
      </div>
    </>
  );
}

function ImportStat({ n, label, tone }: { n: number; label: string; tone: string }) {
  return (
    <div className={`import-stat is-${tone}${n ? '' : ' is-zero'}`}>
      <strong>{n}</strong>
      <span>{label}</span>
    </div>
  );
}
