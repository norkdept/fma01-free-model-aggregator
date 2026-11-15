import type { Json } from './providers.ts';
import {
  DONE_FRAME,
  contentChunk,
  finishChunk,
  finishReasonFromAnthropic,
  parseFrame,
  asRecord,
  sseFrameOf,
  sseLines,
  ssePayload,
  EPOCH,
} from './stream.ts';
import { anthropicToOpenAI, openAIToAnthropic, anthropicEventsFromOpenAI, sseEvent } from './wire.ts';
import { anthropicToGemini, geminiMeta, geminiText, geminiToOpenAI, openAIToGemini } from './gemini.ts';
import type { Wire } from './providers.ts';

/**
 * One body in, one body out, whichever provider wire won the ranking.
 *
 * The handlers decide *who* serves; this module decides *how* to say it in that provider's dialect.
 * Keeping the two apart is what lets the ranking change the provider without the HTTP layer caring,
 * and it is what lets the baseline's openai-only assumption be replaced: an anthropic- or gemini-wire
 * provider is now a first-class answer instead of a 501.
 */

/**
 * The upstream body for a Chat-Completions request that an OpenAI client sent.
 *
 * The client's own fields (tools, response_format, max_tokens) are dropped for wires that cannot carry
 * them; `wireCarries` in the router already refused the ones that would break, so what remains is the
 * translation of shape, not of capability.
 */
export function openAIBodyFor(wire: Wire, body: Json, model: string): Json {
  if (wire === 'gemini') return openAIToGemini(body);
  if (wire === 'anthropic') {
    // Anthropic has no response_format and requires max_tokens.
    const translated = anthropicShapeFromOpenAI(body, model);
    return translated;
  }
  // The client is not required to ask for usage, but a stream without a usage chunk scores invalid, so
  // the hop asks for it on the client's behalf and repairs the tail if the provider still drops it.
  const passthrough: Json = { ...body, model };
  if (body.stream === true) passthrough.stream_options = { include_usage: true };
  return passthrough;
}

/** An OpenAI body re-expressed as a Messages body, for an anthropic-wire provider. */
function anthropicShapeFromOpenAI(body: Json, model: string): Json {
  const record = asRecord(body) ?? {};
  const messages = Array.isArray(record.messages) ? record.messages : [];
  const system: string[] = [];
  const turns: Array<{ role: string; content: unknown }> = [];
  for (const entry of messages) {
    const message = asRecord(entry) ?? {};
    const role = typeof message.role === 'string' ? message.role : 'user';
    if (role === 'system' || role === 'developer') {
      if (typeof message.content === 'string') system.push(message.content);
      continue;
    }
    turns.push({ role: role === 'assistant' ? 'assistant' : 'user', content: message.content ?? '' } as { role: string; content: unknown });
  }
  const out: Record<string, unknown> = {
    model,
    max_tokens: typeof record.max_tokens === 'number' ? record.max_tokens : typeof record.max_completion_tokens === 'number' ? record.max_completion_tokens : 4096,
    messages: turns,
  };
  if (system.length > 0) out.system = system.join('\n\n');
  if (typeof record.temperature === 'number') out.temperature = record.temperature;
  if (typeof record.top_p === 'number') out.top_p = record.top_p;
  if (record.stop !== undefined) out.stop_sequences = typeof record.stop === 'string' ? [record.stop] : record.stop;
  // Streaming is a transport choice the client made; the provider must be told, or it answers whole.
  if (record.stream === true) out.stream = true;
  return out as Json;
}

/** The upstream body for a Messages request that an Anthropic client sent. */
export function anthropicBodyFor(wire: Wire, body: Json, model: string): Json {
  if (wire === 'gemini') return anthropicToGemini(body);
  if (wire === 'anthropic') return { ...body, model };
  return anthropicToOpenAI(body, model);
}

/** An upstream answer from a non-openai wire, expressed as a Chat Completions answer. */
export function answerToOpenAI(wire: Wire, answer: unknown, model: string): Json {
  if (wire === 'gemini') return geminiToOpenAI(answer, model);
  return messagesAnswerToChat(asRecord(answer) ?? {}, model);
}

/**
 * A Messages *answer* re-expressed as a Chat Completions answer.
 *
 * wire.ts's anthropicToOpenAI translates a Messages *request* into a Chat request, which is the other
 * direction and drops the answer's content; an answer needs its `content[]` text blocks flattened into
 * `choices[0].message.content`, which is what this does.
 */
function messagesAnswerToChat(answer: Record<string, unknown>, model: string): Json {
  const blocks = Array.isArray(answer.content) ? answer.content : [];
  let text = '';
  const toolCalls: Json[] = [];
  for (const block of blocks) {
    const item = asRecord(block) ?? {};
    if (item.type === 'text' && typeof item.text === 'string') text += item.text;
    if (item.type === 'tool_use') {
      toolCalls.push({
        id: typeof item.id === 'string' ? item.id : 'toolu_unknown',
        type: 'function',
        function: {
          name: typeof item.name === 'string' ? item.name : '',
          arguments: JSON.stringify(asRecord(item.input) ?? {}),
        },
      });
    }
  }
  const message: Record<string, unknown> = { role: 'assistant', content: text };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  const usage = asRecord(answer.usage) ?? {};
  const finish = answer.stop_reason === 'max_tokens' ? 'length' : toolCalls.length > 0 ? 'tool_calls' : 'stop';
  return {
    id: typeof answer.id === 'string' ? answer.id : `chatcmpl-${model}`,
    object: 'chat.completion',
    created: EPOCH,
    model,
    choices: [{ index: 0, message, finish_reason: finish }],
    usage: {
      prompt_tokens: typeof usage.input_tokens === 'number' ? usage.input_tokens : 0,
      completion_tokens: typeof usage.output_tokens === 'number' ? usage.output_tokens : 0,
      total_tokens: (typeof usage.input_tokens === 'number' ? usage.input_tokens : 0) + (typeof usage.output_tokens === 'number' ? usage.output_tokens : 0),
    },
  } as Json;
}

/** An upstream answer from a non-anthropic wire, expressed as a Messages answer. */
export function answerToAnthropic(wire: Wire, answer: unknown, model: string): Json {
  if (wire === 'openai') return openAIToAnthropic(asRecord(answer) ?? {}, model);
  if (wire === 'gemini') {
    const text = geminiText(answer);
    const meta = geminiMeta(answer);
    return {
      id: `msg_${model}`,
      type: 'message',
      role: 'assistant',
      model,
      content: [{ type: 'text', text }],
      stop_reason: meta.finish === 'length' ? 'max_tokens' : meta.finish === 'stop' ? 'end_turn' : 'end_turn',
      stop_sequence: null,
      usage: { input_tokens: meta.promptTokens, output_tokens: meta.completionTokens },
    } as Json;
  }
  return asRecord(answer) ?? {} as Json;
}

/**
 * Forward an OpenAI-wire SSE stream, repairing the one thing the wire is not reliable about: usage.
 *
 * Providers are asked for `stream_options.include_usage`, but several of the free ones drop it, and a
 * stream that ends with no usage chunk scores the request invalid. The frames are passed through byte
 * for byte; when the stream closes without a usage-bearing chunk, one closing chunk is synthesised so
 * the client's counts stay comparable. `usage` is null once a real usage chunk has been seen.
 */
export async function* openAIEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let sawUsage = false;
  let finish: string | null = null;
  let text = '';
  for await (const line of sseLines(upstream)) {
    const frame = sseFrameOf(line);
    // The provider's own terminator is held back and re-emitted after the usage chunk: anything sent
    // after `[DONE]` is outside the stream, and a usage frame there scores as "events followed [DONE]".
    if (frame.done) continue;
    if (frame.payload !== null) {
      const parsed = parseFrame(frame.payload);
      const record = asRecord(parsed);
      if (record !== null) {
        if (asRecord(record.usage) !== null) sawUsage = true;
        const choices = Array.isArray(record.choices) ? record.choices : [];
        const first = asRecord(choices[0]) ?? {};
        if (typeof first.finish_reason === 'string' && first.finish_reason !== '') finish = first.finish_reason;
        const delta = asRecord(first.delta) ?? {};
        if (typeof delta.content === 'string') text += delta.content;
        else if (typeof delta.reasoning_content === 'string') text += delta.reasoning_content;
      }
    }
    // A frame the provider malformed is forwarded as it stands; only usage is repaired here.
    yield `${line}\n`;
  }
  if (!sawUsage) {
    // No usage chunk arrived. This hop cannot see the provider's prompt count, so it is omitted rather
    // than guessed, and completion tokens are estimated from the text that actually crossed the wire.
    const estimated = Math.max(1, Math.ceil(text.length / 4));
    yield finishChunk(model, finish ?? 'stop', 0, estimated);
  }
  // Terminate unconditionally: the terminator was swallowed above, and a stream without one is scored
  // "the stream did not end with [DONE]" whether or not the provider supplied its own.
  yield DONE_FRAME;
}

export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let finish = 'stop';
  let promptTokens = 0;
  let completionTokens = 0;
  for await (const line of sseLines(upstream)) {
    const frame = sseFrameOf(line);
    if (frame.payload === null || frame.done) continue;
    const event = asRecord(parseFrame(frame.payload));
    if (event === null) continue;
    if (event.type === 'message_start') {
      const usage = asRecord(asRecord(event.message)?.usage) ?? {};
      if (typeof usage.input_tokens === 'number') promptTokens = usage.input_tokens;
      continue;
    }
    if (event.type === 'content_block_delta') {
      const delta = asRecord(event.delta) ?? {};
      if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
        yield contentChunk(model, delta.text);
      }
      continue;
    }
    if (event.type === 'message_delta') {
      const usage = asRecord(event.usage) ?? {};
      if (typeof usage.output_tokens === 'number') completionTokens = usage.output_tokens;
      const mapped = finishReasonFromAnthropic(asRecord(event.delta)?.stop_reason);
      if (mapped !== null) finish = mapped;
    }
  }
  // The finish chunk carries the stop reason and the counts together, the order an openai client expects.
  yield finishChunk(model, finish, promptTokens, completionTokens);
  yield DONE_FRAME;
}

export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let finish = 'stop';
  // Usage rides the last frame; kept so the closing chunk can report the provider's own counts.
  let promptTokens = 0;
  let completionTokens = 0;
  for await (const line of sseLines(upstream)) {
    const frame = sseFrameOf(line);
    if (frame.payload === null || frame.done) continue;
    const parsed = parseFrame(frame.payload);
    const meta = geminiMeta(parsed);
    finish = meta.finish;
    if (meta.promptTokens > 0) promptTokens = meta.promptTokens;
    if (meta.completionTokens > 0) completionTokens = meta.completionTokens;
    const text = geminiText(parsed);
    if (text !== '') yield contentChunk(model, text);
  }
  // A streamed answer still has to report its usage: a stream without a closing usage frame scores the
  // whole request invalid, and the finish chunk is where the openai wire puts it with include_usage.
  yield finishChunk(model, finish, promptTokens, completionTokens);
  yield DONE_FRAME;
}

/** Re-frame any provider stream as Anthropic Messages events. */
export async function* anthropicEventsFrom(upstream: ReadableStream<Uint8Array>, wire: Wire, model: string): AsyncGenerator<string> {
  if (wire === 'openai') {
    yield* anthropicEventsFromOpenAI(upstream, model);
    return;
  }
  if (wire === 'gemini') {
    yield* anthropicEventsFromGemini(upstream, model);
    return;
  }
  // An anthropic-wire provider already speaks the event names the client wants.
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      yield decoder.decode(value, { stream: true });
    }
  } finally {
    reader.releaseLock();
  }
}

/** A gemini stream re-expressed as the Anthropic event order. */
async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  yield sseEvent('message_start', {
    type: 'message_start',
    message: { id: `msg_${model}`, type: 'message', role: 'assistant', model, content: [], stop_reason: null, usage: { input_tokens: 0, output_tokens: 0 } },
  });
  yield sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  let finish = 'end_turn';
  // The upstream usage arrives on the closing openai chunk; hold it until message_delta, which is
  // where the Anthropic wire reports output tokens.
  let inputTokens = 0;
  let outputTokens = 0;
  for await (const frame of openAIEventsFromGemini(upstream, model)) {
    if (!frame.startsWith('data: ') || frame.includes('[DONE]')) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(frame.slice(6).trim());
    } catch {
      continue;
    }
    const record = asRecord(parsed) ?? {};
    const usage = asRecord(record.usage) ?? {};
    if (typeof usage.prompt_tokens === 'number') inputTokens = usage.prompt_tokens;
    if (typeof usage.completion_tokens === 'number') outputTokens = usage.completion_tokens;
    const choices = Array.isArray(record.choices) ? record.choices : [];
    const first = asRecord(choices[0]) ?? {};
    const delta = asRecord(first.delta) ?? {};
    if (typeof delta.content === 'string' && delta.content !== '') {
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: delta.content } });
    }
    if (typeof first.finish_reason === 'string' && first.finish_reason !== '') {
      finish = first.finish_reason === 'length' ? 'max_tokens' : 'end_turn';
    }
  }
  yield sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: finish, stop_sequence: null }, usage: { input_tokens: inputTokens, output_tokens: outputTokens } });
  yield sseEvent('message_stop', { type: 'message_stop' });
}
