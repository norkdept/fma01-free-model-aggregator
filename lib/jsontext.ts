/**
 * Pulling JSON documents out of text that is not one document.
 *
 * The provider directory arrives as a body that holds several top-level objects in sequence rather than
 * a single JSON value, so it cannot be handed to `JSON.parse` as it stands. Scanning for it was inline in
 * providers.ts, next to the code that interprets the result, and the scanner had two faults: it restarted
 * a scan at every `{` including the ones nested inside an object it had already matched, and it treated
 * `\"` as escaping the next character of the *text* rather than as part of the string, which is right for
 * a quote and wrong for a backslash that is itself escaped. Both are fixed here, and the reading of the
 * values lives with the scanning rather than beside the interpretation.
 */

/** A plain object, the only shape a directory entry may have. */
export type JsonRecord = Record<string, unknown>;

/** True when a parsed value is a plain object rather than an array or a scalar. */
export function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * The index of the brace closing the object that opens at `open`, or -1 when it never closes.
 *
 * String contents are skipped, so a `}` inside a value does not end the object. The escape handling
 * advances past the escaped character rather than only past the backslash: for `\\"` the backslash
 * escapes the second backslash, leaving the quote to close the string, and a scanner that stopped at
 * the backslash would read the rest of the text as still inside a string.
 */
export function closingBrace(text: string, open: number): number {
  let depth = 0;
  let inString = false;
  for (let i = open; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/**
 * Every object in `text` that parses as JSON and carries a string `schema`, in the order they appear.
 *
 * Each match skips the text it consumed, so an object nested inside a matched one is not rescanned as a
 * candidate of its own: the outer object already carries it, and rescanning was quadratic on a large
 * directory. An object that does not parse is not a match, but the scan resumes one character past its
 * opening brace rather than past its end, since the failure may be a brace inside a value rather than
 * the object boundary.
 */
export function documentsWithSchema(text: string, field = 'schema'): JsonRecord[] {
  const found: JsonRecord[] = [];
  let open = text.indexOf('{');
  while (open !== -1) {
    const close = closingBrace(text, open);
    if (close === -1) break;
    let matched: JsonRecord | null = null;
    try {
      const value: unknown = JSON.parse(text.slice(open, close + 1));
      if (isRecord(value) && typeof value[field] === 'string') matched = value;
    } catch {
      // Not a document on its own; the scan moves past this brace and looks on.
    }
    if (matched !== null) {
      found.push(matched);
      open = text.indexOf('{', close + 1);
    } else {
      open = text.indexOf('{', open + 1);
    }
  }
  return found;
}

/** Read a value as a non-empty string, or the empty string when it is anything else. */
export function stringOrEmpty(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

/** Read a value as a finite number, or null when it is not one. NaN and Infinity are not numbers here. */
export function finiteOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** Read a value as a boolean, or null when it is not one; a truthy string is not a boolean. */
export function booleanOrNull(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

/**
 * Read a value as a list of objects, dropping anything that is not one.
 *
 * A directory that lists models as arrays of objects is read leniently: one malformed entry must not
 * discard the entries beside it, since a provider that answers is more useful than one that is skipped
 * over a sibling's bad field.
 */
export function recordsIn(value: unknown): JsonRecord[] {
  if (!Array.isArray(value)) return [];
  const out: JsonRecord[] = [];
  for (const entry of value) if (isRecord(entry)) out.push(entry);
  return out;
}
