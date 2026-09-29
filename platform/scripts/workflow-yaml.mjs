// A small reader for the GitHub workflow files of this repository, so tests and
// checks can look at what a step runs and which paths it names without a YAML
// dependency (these scripts use Node's own modules only, as the CI lanes that
// run them install nothing first).
//
// It reads the YAML the workflows are written in, not YAML in general: block
// mappings and sequences (including `- key: value` items), plain, 'single' and
// "double" quoted scalars, `|` and `>` block scalars, one-line [flow, lists] and
// {flow: maps}, and comments. Values stay strings ("false" is not a boolean).
// What it does not read (anchors, tags, a scalar continued on the next line) it
// refuses, naming the line, rather than reading it wrongly.
//
// A parsed mapping carries, out of sight (non-enumerable), `lines`: the 1-based
// line each key is on, and `content`: for a block scalar, the line its first line
// of text is on.

const hidden = (value, name, data) => Object.defineProperty(value, name, { value: data, enumerable: false });

const KEY = /^(?:"((?:[^"\\]|\\.)*)"|'((?:[^']|'')*)'|([^\s#'"\[\]{}&*!|>%@`,-][^:#]*|-[^\s:][^:#]*))[ \t]*:(?:[ \t]+(.*))?$/;

/** The text of `rest` up to a comment that starts a word (` #`), leaving `#` inside quotes and words alone. */
function withoutComment(rest) {
  let quote = null;
  for (let i = 0; i < rest.length; i++) {
    const c = rest[i];
    if (quote) {
      if (c === '\\' && quote === '"') i++;
      else if (c === quote && quote === "'" && rest[i + 1] === "'") i++; // '' is a quote inside a 'single-quoted' string
      else if (c === quote) quote = null;
    } else if ((c === '"' || c === "'") && (i === 0 || /[\s[{,:]/.test(rest[i - 1]))) quote = c;
    else if (c === '#' && (i === 0 || /\s/.test(rest[i - 1]))) return rest.slice(0, i).trimEnd();
  }
  return rest.trimEnd();
}

/** Parse YAML written as the workflows are. */
export function parseYaml(source) {
  const lines = source.replace(/\r\n?/g, '\n').split('\n');
  let at = 0;
  const fail = (message) => {
    throw new Error(`line ${at + 1}: ${message}`);
  };
  const indentOf = (line) => line.length - line.trimStart().length;
  const isBlank = (line) => /^\s*(?:#.*)?$/.test(line);
  const skipBlank = () => {
    while (at < lines.length && isBlank(lines[at])) at++;
  };

  function scalar(raw) {
    const text = withoutComment(raw.trim());
    if (text.startsWith('"')) {
      const end = /^"((?:[^"\\]|\\.)*)"$/.exec(text);
      if (!end) fail(`a double-quoted scalar has to end on its own line: ${text}`);
      return end[1].replace(/\\(["\\/nrt])/g, (_, c) => ({ n: '\n', r: '\r', t: '\t' })[c] ?? c);
    }
    if (text.startsWith("'")) {
      const end = /^'((?:[^']|'')*)'$/.exec(text);
      if (!end) fail(`a single-quoted scalar has to end on its own line: ${text}`);
      return end[1].replaceAll("''", "'");
    }
    if (/^[&*!]/.test(text)) fail(`anchors, aliases and tags are not read: ${text}`);
    return text;
  }

  /** The items of a one-line flow sequence or the entries of a flow mapping. */
  function flow(text) {
    const closing = text[0] === '[' ? ']' : '}';
    const body = text.slice(1, text.lastIndexOf(closing));
    if (text.lastIndexOf(closing) < 0 || withoutComment(text.slice(text.lastIndexOf(closing) + 1)) !== '') fail(`a flow collection has to end on its own line: ${text}`);
    const parts = [];
    let depth = 0;
    let quote = null;
    let start = 0;
    for (let i = 0; i <= body.length; i++) {
      const c = body[i];
      if (quote) {
        if (c === quote) quote = null;
      } else if (c === '"' || c === "'") quote = c;
      else if (c === '[' || c === '{') depth++;
      else if (c === ']' || c === '}') depth--;
      else if ((c === ',' && depth === 0) || i === body.length) {
        if (body.slice(start, i).trim() !== '') parts.push(body.slice(start, i).trim());
        start = i + 1;
      }
    }
    if (closing === ']') return parts.map((part) => (/^[[{]/.test(part) ? flow(part) : scalar(part)));
    const map = {};
    for (const part of parts) {
      const at2 = part.indexOf(':');
      if (at2 < 0) fail(`a flow mapping entry needs a colon: ${part}`);
      const value = part.slice(at2 + 1).trim();
      map[scalar(part.slice(0, at2))] = /^[[{]/.test(value) ? flow(value) : scalar(value);
    }
    return map;
  }

  function value(rest, indent, key, meta) {
    const text = withoutComment(rest);
    if (/^[|>][+-]?[0-9]?$/.test(text)) {
      const keyLine = at + 1;
      at++;
      const out = [];
      let contentIndent = null;
      let first = null;
      while (at < lines.length) {
        const line = lines[at];
        if (line.trim() === '') {
          out.push('');
          at++;
          continue;
        }
        const own = indentOf(line);
        if (own <= indent) break;
        if (contentIndent === null) {
          contentIndent = own;
          first = at + 1;
        }
        if (own < contentIndent) fail('a block scalar line is indented less than the first');
        out.push(line.slice(contentIndent));
        at++;
      }
      while (out.length && out[out.length - 1] === '') out.pop();
      const fold = text[0] === '>';
      let body = fold ? out.join(' ') : out.join('\n');
      if (!text.includes('-')) body += '\n';
      meta.content[key] = first ?? keyLine;
      return body;
    }
    if (text === '') {
      at++;
      skipBlank();
      if (at < lines.length) {
        const own = indentOf(lines[at]);
        if (own > indent) return block(own);
        if (own === indent && /^-( |$)/.test(lines[at].trimStart())) return sequence(indent);
      }
      return null;
    }
    const parsed = text[0] === '[' || text[0] === '{' ? flow(text) : scalar(text);
    at++;
    return parsed;
  }

  function mapping(indent) {
    const map = {};
    const meta = { lines: {}, content: {} };
    hidden(map, 'lines', meta.lines);
    hidden(map, 'content', meta.content);
    for (;;) {
      skipBlank();
      if (at >= lines.length) break;
      const own = indentOf(lines[at]);
      if (own < indent) break;
      if (own > indent) fail('unexpected indentation');
      const found = KEY.exec(lines[at].slice(indent));
      if (!found) fail(`not a "key: value" line: ${lines[at].trim()}`);
      const key = found[1] ?? found[2]?.replaceAll("''", "'") ?? found[3];
      meta.lines[key] = at + 1;
      map[key] = value(found[4] ?? '', indent, key, meta);
    }
    return map;
  }

  function sequence(indent) {
    const list = [];
    hidden(list, 'lines', []);
    for (;;) {
      skipBlank();
      if (at >= lines.length || indentOf(lines[at]) !== indent || !/^-( |$)/.test(lines[at].trimStart())) break;
      list.lines.push(at + 1);
      const after = lines[at].trimStart().slice(1).replace(/^ /, '');
      if (withoutComment(after) === '') {
        at++;
        skipBlank();
        list.push(at < lines.length && indentOf(lines[at]) > indent ? block(indentOf(lines[at])) : null);
      } else if (KEY.test(withoutComment(after))) {
        lines[at] = ' '.repeat(indent + 2) + after;
        list.push(mapping(indent + 2));
      } else {
        const text = withoutComment(after);
        list.push(text[0] === '[' || text[0] === '{' ? flow(text) : scalar(text));
        at++;
      }
    }
    return list;
  }

  function block(indent) {
    skipBlank();
    if (at >= lines.length) return null;
    return /^-( |$)/.test(lines[at].trimStart()) ? sequence(indent) : mapping(indent);
  }

  skipBlank();
  const root = at < lines.length ? block(indentOf(lines[at])) : null;
  skipBlank();
  if (at < lines.length) fail(`unexpected content: ${lines[at].trim()}`);
  return root;
}

/**
 * Every step of every job of a parsed workflow, as
 * `{ job, jobLine, step, line, keyLines, name, id, uses, run, runLine,
 *    workingDirectory, defaultDirectory, env, with, withLines, withContent }`: `line`
 * is the step's first line, `keyLines` the line of each of its keys, `runLine`
 * the line of the first line of its script. A job that calls a reusable workflow
 * (`uses:` on the job) is one step of its own with `step: null`.
 */
export function workflowSteps(document) {
  const steps = [];
  for (const [job, spec] of Object.entries(document?.jobs ?? {})) {
    const jobLine = document.jobs.lines[job];
    const defaultDirectory = spec.defaults?.run?.['working-directory'] ?? '';
    if (spec.uses) steps.push({ job, jobLine, step: null, line: spec.lines.uses, keyLines: spec.lines, name: `calls ${spec.uses}`, uses: spec.uses, with: spec.with ?? {}, withLines: spec.with?.lines ?? {}, defaultDirectory });
    (spec.steps ?? []).forEach((step, index) => {
      steps.push({
        job,
        jobLine,
        step: index,
        line: spec.steps.lines[index],
        keyLines: step.lines ?? {},
        name: step.name ?? step.id ?? step.uses ?? `step ${index + 1}`,
        id: step.id,
        uses: step.uses,
        run: step.run,
        runLine: step.content?.run ?? step.lines?.run,
        workingDirectory: step['working-directory'],
        defaultDirectory,
        env: step.env ?? {},
        with: step.with ?? {},
        withLines: step.with?.lines ?? {},
        withContent: step.with?.content ?? {},
      });
    });
  }
  return steps;
}
