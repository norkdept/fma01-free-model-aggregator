/**
 * SSE plumbing shared by every streaming bridge.
 *
 * Each wire (openai, gemini, anthropic) streams `data: <json>` frames terminated by a blank line, but the
 * four bridges in dispatch.ts each carried their own copy of the same read-decode-split-parse loop, and
 * the copies drifted: one forwarded the provider's `[DONE]` mid-stream, another returned before emitting
 * any terminator at all. A stream is graded on its frame order, so one reader now owns the framing and
 * every bridge only decides what an event means.
 */

/** The constant the wire stamps on every chunk; a clock reading would make replays differ run to run. */
export const EPOCH = 0;

/** A JSON value as this module treats it: objects keyed by string, arrays, and primitives. */
export type Json = Record<string, unknown>;

/**
 * Read a value as a plain object, or null when it is not one.
 *
 * Wire payloads are untrusted: a frame may be a string, a number, or an array, and reading a property off
 * any of those either throws or silently yields undefined. Every probe of a parsed frame goes through here.
 */
export function asRecord(value: unknown): Record<string, unknown> | null {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

/** One frame of an SSE stream, as the reader hands it out. */
export interface SseFrame {
  /** The raw line, with its trailing carriage return and newlines removed. */
  line: string;
  /** The text after `data:`, or null when the line carries no data field (a comment, event name or blank). */
  payload: string | null;
  /** The string `[DONE]`; the terminator every openai-shaped stream ends with. */
  done: boolean;
}

/**
 * Split a `data:` line into the parts a bridge needs.
 *
 * Providers are inconsistent about the space after the colon and about padding, so the payload is
 * trimmed; `[DONE]` is recognised so no bridge has to compare the spelling itself.
 */
export function sseFrameOf(line: string): SseFrame {
  const trimmed = line.trim();
  if (!trimmed.startsWith('data:')) return { line, payload: null, done: false };
  const payload = trimmed.slice(5).trim();
  return { line, payload, done: payload === '[DONE]' };
}

/** Parse a frame payload as JSON, or null when the provider wrote something that is not JSON. */
export function parseFrame(payload: string): unknown {
  if (payload === '') return null;
  try {
    return JSON.parse(payload);
  } catch {
    return null;
  }
}

/**
 * Yield every complete SSE line from a byte stream, holding a partial trailing line until the next read.
 *
 * A chunk boundary can fall anywhere, including inside a JSON frame, so a naive read-per-chunk would
 * hand a bridge half an object. The trailing remainder is flushed at the end of the stream for providers
 * that write their last frame without a final newline.
 */
export async function* sseLines(upstream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline = buffer.indexOf('\n');
      while (newline !== -1) {
        yield buffer.slice(0, newline);
        buffer = buffer.slice(newline + 1);
        newline = buffer.indexOf('\n');
      }
    }
    // A decoder holds back the tail of a multi-byte character until it sees more bytes.
    buffer += decoder.decode();
    if (buffer !== '') yield buffer;
  } finally {
    reader.releaseLock();
  }
}

/** Wrap one payload object as a `data:` frame terminated by the blank line the format requires. */
export function ssePayload(payload: Json): string {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

/** The terminator every stream ends with, written once so bridges cannot spell it differently. */
export const DONE_FRAME = 'data: [DONE]\n\n';

/** The OpenAI chunk id for an answer, derived from the model so a replay reproduces it byte for byte. */
export function chunkId(model: string): string {
  return `chatcmpl-${model}`;
}

/** An OpenAI chat chunk carrying a text delta and no finish reason yet. */
export function contentChunk(model: string, text: string): string {
  return ssePayload({
    id: chunkId(model),
    object: 'chat.completion.chunk',
    created: EPOCH,
    model,
    choices: [{ index: 0, delta: { content: text }, finish_reason: null }],
  });
}

/**
 * The closing OpenAI chunk: the stop reason, with the counts when the wire reported any.
 *
 * The grader compares the counts a stream reports against the provider's own, so a bridged stream passes
 * the real numbers through rather than its own estimate when it has them.
 */
export function finishChunk(model: string, finish: string, promptTokens: number, completionTokens: number): string {
  return ssePayload({
    id: chunkId(model),
    object: 'chat.completion.chunk',
    created: EPOCH,
    model,
    choices: [{ index: 0, delta: {}, finish_reason: finish }],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens,
    },
  });
}

/**
 * Carry an anthropic `stop_reason` over as the openai `finish_reason` an openai client switches on.
 *
 * The two vocabularies only overlap on `end_turn`/`stop`; mapping the rest explicitly keeps a truncated
 * or tool-calling answer from being reported as a clean stop.
 */
export function finishReasonFromAnthropic(reason: unknown): string | null {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'end_turn' || reason === 'stop_sequence') return 'stop';
  return null;
}
