/**
 * Reading a request body off the socket, with the failures named.
 *
 * The server used to answer a single `null` for every bad body: unreadable JSON, a body over the cap,
 * and a body that was valid JSON but not an object all collapsed into one empty answer. A client that
 * sent 40 MiB got the same shape of reply as one that fat-fingered a comma, and a tool_use payload that
 * an SDK could not round-trip was indistinguishable from a transport problem. The caller cannot pick a
 * status to answer with when the reader tells it nothing, so the reader says which failure it was.
 */
import type { IncomingMessage } from 'node:http';

/** What is wrong with a body, in the terms an HTTP status is chosen from. */
export type BodyFailure = 'empty' | 'too_large' | 'unreadable' | 'not_object';

/** A body that was read, or the reason it was not. */
export type BodyResult = { ok: true; body: Record<string, unknown> } | { ok: false; reason: BodyFailure };

/** A JSON object as the handlers treat it: wire bodies are objects, never arrays or bare scalars. */
export type JsonObject = Record<string, unknown>;

/** True when a parsed value is a plain object, the only shape a request body may have. */
export function isJsonObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * The status a client should be told for each way a body can be refused.
 *
 * 413 for the size cap, 400 for anything the client can fix by sending different bytes: the distinction
 * matters because an SDK retries a 413 differently from a 400.
 */
export function statusForBodyFailure(reason: BodyFailure): number {
  return reason === 'too_large' ? 413 : 400;
}

/**
 * The message an error body carries, written per reason so a log names the actual fault.
 */
export function describeBodyFailure(reason: BodyFailure, cap: number): string {
  switch (reason) {
    case 'empty':
      return 'the request body was empty';
    case 'too_large':
      return `the request body was larger than ${cap} bytes`;
    case 'unreadable':
      return 'the request body was not readable as JSON';
    case 'not_object':
      return 'the request body was JSON but not an object';
  }
}

/**
 * Read the whole body, refusing it once it passes the cap.
 *
 * The size is checked as the chunks arrive rather than after the read finishes, so an oversized upload
 * stops being buffered instead of exhausting the process first. A body that crosses the cap is drained
 * to the end anyway: answering before the client has finished writing resets the connection, and the
 * client then sees a broken pipe instead of the 413 that explains what happened.
 */
export async function readBody(req: IncomingMessage, cap: number): Promise<{ result: BodyResult; bytes: number }> {
  const chunks: Buffer[] = [];
  let size = 0;
  let oversized = false;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > cap) {
      oversized = true;
      // Hold nothing more: the reply is already decided, and keeping the tail would double the waste.
      continue;
    }
    chunks.push(buffer);
  }
  if (oversized) return { result: { ok: false, reason: 'too_large' }, bytes: size };
  // An empty body arrives as no chunks or as a single zero-length one; both are the same fault, and
  // checking the length rather than the chunk count names it correctly either way.
  if (size === 0) return { result: { ok: false, reason: 'empty' }, bytes: 0 };
  let value: unknown;
  try {
    value = JSON.parse(Buffer.concat(chunks).toString('utf8'));
  } catch {
    return { result: { ok: false, reason: 'unreadable' }, bytes: size };
  }
  if (!isJsonObject(value)) return { result: { ok: false, reason: 'not_object' }, bytes: size };
  return { result: { ok: true, body: value }, bytes: size };
}

/**
 * Pick the field a request names its own size limit with, across the two wire dialects.
 *
 * OpenAI clients send `max_tokens` (and json-mode clients `max_completion_tokens`); Anthropic clients
 * send `max_tokens` too but require it. Both are reported so a refusal can name what the client asked.
 */
export function requestedMaxTokens(body: JsonObject): number | null {
  const candidates = [body.max_completion_tokens, body.max_tokens];
  for (const value of candidates) {
    if (typeof value === 'number' && Number.isFinite(value) && value > 0) return Math.floor(value);
  }
  return null;
}
