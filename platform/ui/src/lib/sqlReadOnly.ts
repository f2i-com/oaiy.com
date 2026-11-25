/**
 * Conservative SQLite read-only classifier used by DataViewer's read-only guard.
 *
 * Parses comments and string/identifier literals properly so that:
 * - Multi-statement queries (`;` outside literals after stripping at most one
 *   trailing `;`) are always rejected.
 * - Writable CTEs (`WITH ... INSERT/UPDATE/DELETE/REPLACE/DROP/ALTER/ATTACH/VACUUM/...`)
 *   are rejected.
 * - `PRAGMA` is restricted to parameterless reads (rejects `=` and `(`).
 * - `EXPLAIN [QUERY PLAN]` is allowed only when the explained statement itself is
 *   read-only.
 * - Unclosed quotes or block comments fail closed (return `false`).
 */

interface StrippedSql {
  /** Comments replaced with spaces; quoted string/identifier bodies replaced with neutral tokens. */
  sanitized: string;
  /** True if a string literal, quoted identifier, or block comment was never closed. */
  unclosed: boolean;
}

function stripSqlCommentsAndLiterals(sql: string): StrippedSql {
  let out = '';
  let i = 0;
  const len = sql.length;

  while (i < len) {
    const ch = sql[i];

    // Single-quoted string literal ('...' with '' escape)
    if (ch === "'") {
      i++;
      let closed = false;
      while (i < len) {
        if (sql[i] === "'") {
          i++;
          if (i < len && sql[i] === "'") {
            i++;
            continue;
          }
          closed = true;
          break;
        }
        i++;
      }
      if (!closed) return { sanitized: out, unclosed: true };
      out += " '' ";
      continue;
    }

    // Double-quoted identifier ("..." with "" escape)
    if (ch === '"') {
      i++;
      let closed = false;
      while (i < len) {
        if (sql[i] === '"') {
          i++;
          if (i < len && sql[i] === '"') {
            i++;
            continue;
          }
          closed = true;
          break;
        }
        i++;
      }
      if (!closed) return { sanitized: out, unclosed: true };
      out += ' "" ';
      continue;
    }

    // Backtick-quoted identifier (`...` with `` escape)
    if (ch === '`') {
      i++;
      let closed = false;
      while (i < len) {
        if (sql[i] === '`') {
          i++;
          if (i < len && sql[i] === '`') {
            i++;
            continue;
          }
          closed = true;
          break;
        }
        i++;
      }
      if (!closed) return { sanitized: out, unclosed: true };
      out += ' `` ';
      continue;
    }

    // Bracket-quoted identifier ([...])
    if (ch === '[') {
      i++;
      while (i < len && sql[i] !== ']') i++;
      if (i >= len) return { sanitized: out, unclosed: true };
      i++;
      out += ' [] ';
      continue;
    }

    // Line comment (-- ... \n)
    if (ch === '-' && i + 1 < len && sql[i + 1] === '-') {
      i += 2;
      while (i < len && sql[i] !== '\n') i++;
      out += ' ';
      continue;
    }

    // Block comment (/* ... */)
    if (ch === '/' && i + 1 < len && sql[i + 1] === '*') {
      i += 2;
      let closed = false;
      while (i + 1 < len) {
        if (sql[i] === '*' && sql[i + 1] === '/') {
          i += 2;
          closed = true;
          break;
        }
        i++;
      }
      if (!closed) return { sanitized: out, unclosed: true };
      out += ' ';
      continue;
    }

    out += ch;
    i++;
  }

  return { sanitized: out, unclosed: false };
}

const CTE_FORBIDDEN_KEYWORDS = new Set([
  'insert',
  'update',
  'delete',
  'replace',
  'drop',
  'alter',
  'attach',
  'detach',
  'vacuum',
  'create',
  'reindex',
  'pragma',
  'begin',
  'commit',
  'rollback',
  'savepoint',
  'release',
]);

function checkSanitizedReadOnly(body: string, allowExplain: boolean): boolean {
  const trimmed = body.trim();
  if (!trimmed) return false;

  const lower = trimmed.toLowerCase();
  const firstWordMatch = /^([a-z_][a-z0-9_]*)/.exec(lower);
  if (!firstWordMatch) return false;
  const firstWord = firstWordMatch[1];

  if (firstWord === 'select') {
    return true;
  }

  if (firstWord === 'with') {
    const words: string[] = lower.match(/[a-z_][a-z0-9_]*/g) ?? [];
    if (!words.includes('select') && !words.includes('values')) return false;
    for (const w of words) {
      if (CTE_FORBIDDEN_KEYWORDS.has(w)) return false;
    }
    return true;
  }

  if (firstWord === 'pragma') {
    // Parameterless PRAGMA reads only (reject `=` assignments and `(...)` calls).
    if (lower.includes('=') || lower.includes('(') || lower.includes(')')) {
      return false;
    }
    return /^pragma\s+[a-z_][a-z0-9_]*(?:\s*\.\s*[a-z_][a-z0-9_]*)?$/.test(lower);
  }

  if (allowExplain && firstWord === 'explain') {
    const rest = trimmed
      .replace(/^explain\s+(?:query\s+plan\s+)?/i, '')
      .trim();
    if (!rest || rest === trimmed) return false;
    return checkSanitizedReadOnly(rest, false);
  }

  return false;
}

/**
 * Classify an arbitrary SQL string as strictly read-only (`true`) or potentially
 * mutating/multi-statement (`false`).
 */
export function isReadOnlySql(sql: string): boolean {
  if (typeof sql !== 'string') return false;
  const { sanitized, unclosed } = stripSqlCommentsAndLiterals(sql);
  if (unclosed) return false;

  // Allow at most a single trailing semicolon; reject any other semicolon outside literals.
  let body = sanitized.trim();
  if (body.endsWith(';')) {
    body = body.slice(0, -1).trim();
  }
  if (!body || body.includes(';')) return false;

  return checkSanitizedReadOnly(body, true);
}
