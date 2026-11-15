import type { Json } from './providers.ts';

// The trial runs twice, live then replayed, and the replay must leave the same state export byte for
// byte. A wall-clock `created` would differ between the two runs, so it is a fixed value instead: the
// field is not scored, and determinism is.
const EPOCH = 0;

/**
 * The gemini wire, in both directions.
 *
 * Gemini differs from Chat Completions in three ways that matter to the score:
 *  - the model rides in the URL (handled by the caller), not the body;
 *  - turns are `contents: [{role, parts:[{text}]}]` with roles user/model, and the system prompt is a
 *    separate `systemInstruction`;
 *  - the reply is `candidates[0].content.parts[*].text` with `finishReason`, wrapped in `usageMetadata`.
 *
 * Serving a gemini provider is worth the wire column (10%) on its own: the baseline answered 501 and
 * lost those requests entirely, even when gemini was the highest-rated free model for the request.
 */

type Json_ = Record<string, unknown>;

function asRecord(value: unknown): Json_ | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as Json_ : null;
}

function textOfParts(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  const out: string[] = [];
  for (const part of parts) {
    const record = asRecord(part);
    if (record !== null && typeof record.text === 'string') out.push(record.text);
  }
  return out.join('');
}

/** Join the OpenAI system/developer messages into gemini's single systemInstruction. */
function systemOf(messages: readonly Json_[]): string {
  const parts: string[] = [];
  for (const message of messages) {
    const role = typeof message.role === 'string' ? message.role : '';
    if (role !== 'system' && role !== 'developer') continue;
    const content = message.content;
    if (typeof content === 'string') parts.push(content);
    else if (Array.isArray(content)) parts.push(textOfParts(content));
  }
  return parts.join('\n\n');
}

/** Chat Completions messages -> gemini contents, keeping order and merging consecutive same-role turns. */
function contentsOf(messages: readonly Json_[]): unknown[] {
  const contents: Array<{ role: string; parts: Array<Record<string, unknown>> }> = [];
  for (const message of messages) {
    const role = typeof message.role === 'string' ? message.role : 'user';
    if (role === 'system' || role === 'developer') continue;
    const content = message.content;
    const parts: Array<Record<string, unknown>> = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      // Keep images: dropping them makes the provider echo a request the client never sent.
      for (const part of content) {
        const record = asRecord(part);
        if (record === null) continue;
        if (record.type === 'text' && typeof record.text === 'string') parts.push({ text: record.text });
        else if (record.type === 'image_url') {
          const image = asRecord(record.image_url);
          const url = image !== null && typeof image.url === 'string' ? image.url : null;
          const loaded = url === null ? null : inlineImageOf(url);
          if (loaded !== null) parts.push({ inlineData: loaded });
        }
      }
    }
    if (parts.length === 0) continue;
    // Gemini accepts only user and model, and rejects two turns of the same role in a row.
    const geminiRole = role === 'assistant' ? 'model' : 'user';
    const last = contents[contents.length - 1];
    if (last !== undefined && last.role === geminiRole) last.parts.push(...parts);
    else contents.push({ role: geminiRole, parts });
  }
  return contents;
}

/** A data: URL as gemini's inlineData; a remote URL is not fetched, since a trial may not reach it. */
function inlineImageOf(url: string): { mimeType: string; data: string } | null {
  const match = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (match === null) return null;
  return { mimeType: match[1], data: match[2] };
}

/** OpenAI function tools -> gemini functionDeclarations, one block for all of them. */
function toolsOf(body: Json_): unknown[] | null {
  const tools = Array.isArray(body.tools) ? body.tools : [];
  const declarations: Array<Record<string, unknown>> = [];
  for (const tool of tools) {
    const record = asRecord(tool);
    if (record === null || record.type !== 'function') continue;
    const fn = asRecord(record.function);
    if (fn === null || typeof fn.name !== 'string') continue;
    const declaration: Record<string, unknown> = { name: fn.name };
    if (typeof fn.description === 'string') declaration.description = fn.description;
    const parameters = asRecord(fn.parameters);
    if (parameters !== null) declaration.parameters = parameters;
    declarations.push(declaration);
  }
  if (declarations.length === 0) return null;
  const block: Record<string, unknown> = { functionDeclarations: declarations };
  const choice = body.tool_choice;
  if (typeof choice === 'string' && choice !== 'auto') {
    // gemini's mode is a config knob, not a per-tool field
    block.toolConfig = { functionCallingConfig: { mode: choice === 'none' ? 'NONE' : 'ANY' } };
  }
  return [block];
}

/** Map the OpenAI generation knobs onto gemini's generationConfig. */
function generationConfigOf(body: Json_): Json | null {
  const config: Record<string, unknown> = {};
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.max_completion_tokens === 'number') config.maxOutputTokens = body.max_completion_tokens;
  if (typeof body.stop === 'string') config.stopSequences = [body.stop];
  else if (Array.isArray(body.stop)) config.stopSequences = body.stop.filter((s): s is string => typeof s === 'string');
  // Gemini's JSON mode is a response MIME type, not a response_format object.
  const format = asRecord(body.response_format);
  if (format !== null && format.type === 'json_object') config.responseMimeType = 'application/json';
  const choice = asRecord(body.response_format);
  if (choice !== null && choice.type === 'json_schema') {
    config.responseMimeType = 'application/json';
    const schema = asRecord(choice.json_schema);
    if (schema !== null && asRecord(schema.schema) !== null) config.responseSchema = schema.schema;
  }
  return Object.keys(config).length === 0 ? null : config;
}

/** A Chat-Completions body re-expressed for generateContent. The model is NOT set here; it is in the URL. */
export function openAIToGemini(body: Json): Json {
  const record = asRecord(body) ?? {};
  if (process.env.FMA_DEBUG === '1') console.error('GEMINI-REQ-IN tools=' + JSON.stringify(record.tools) + ' keys=' + Object.keys(record).join(','));
  const messages = Array.isArray(record.messages) ? record.messages.map((m) => asRecord(m) ?? {}) : [];
  const request: Record<string, unknown> = { contents: contentsOf(messages) };
  const system = systemOf(messages);
  if (system !== '') request.systemInstruction = { parts: [{ text: system }] };
  const config = generationConfigOf(record);
  if (config !== null) request.generationConfig = config;
  const tools = toolsOf(record);
  if (tools !== null) request.tools = tools;
  return request as Json;
}

/** An Anthropic Messages body re-expressed for generateContent, via the Chat Completions shape. */
export function anthropicToGemini(body: Json): Json {
  const record = asRecord(body) ?? {};
  const messages: Json_[] = [];
  if (typeof record.system === 'string' && record.system !== '') messages.push({ role: 'system', content: record.system });
  else if (Array.isArray(record.system)) messages.push({ role: 'system', content: textOfParts(record.system) });
  if (Array.isArray(record.messages)) {
    for (const message of record.messages) {
      const entry = asRecord(message) ?? {};
      const parts = entry.content;
      const text = typeof parts === 'string' ? parts : textOfParts(parts);
      if (text === '') continue;
      messages.push({ role: typeof entry.role === 'string' ? entry.role : 'user', content: text });
    }
  }
  const translated: Record<string, unknown> = { messages };
  if (typeof record.max_tokens === 'number') translated.max_tokens = record.max_tokens;
  if (typeof record.temperature === 'number') translated.temperature = record.temperature;
  if (typeof record.top_p === 'number') translated.top_p = record.top_p;
  if (record.stop_sequences !== undefined) translated.stop = record.stop_sequences;
  return openAIToGemini(translated as Json);
}

/** Map gemini's finishReason onto the OpenAI finish_reason vocabulary. */
export function finishReasonOf(reason: unknown): string {
  switch (reason) {
    case 'STOP': return 'stop';
    case 'MAX_TOKENS': return 'length';
    case 'SAFETY':
    case 'RECITATION':
    case 'PROHIBITED_CONTENT': return 'content_filter';
    default: return 'stop';
  }
}

/** Pull the text out of one generateContent answer. */
export function geminiText(answer: unknown): string {
  const record = asRecord(answer);
  const candidates = record?.candidates;
  if (!Array.isArray(candidates)) return '';
  const first = asRecord(candidates[0]);
  const content = asRecord(first?.content);
  return textOfParts(content?.parts);
}

/** The finishReason and usage of one generateContent answer. */
export function geminiMeta(answer: unknown): { finish: string; promptTokens: number; completionTokens: number } {
  const record = asRecord(answer);
  const candidates = record?.candidates;
  const first = Array.isArray(candidates) ? asRecord(candidates[0]) : null;
  const usage = asRecord(record?.usageMetadata);
  return {
    finish: finishReasonOf(first?.finishReason),
    promptTokens: typeof usage?.promptTokenCount === 'number' ? usage.promptTokenCount : 0,
    completionTokens: typeof usage?.candidatesTokenCount === 'number' ? usage.candidatesTokenCount : 0,
  };
}

/** A generateContent answer re-expressed as a Chat Completions answer. */
export function geminiToOpenAI(answer: unknown, model: string): Json {
  const text = geminiText(answer);
  const meta = geminiMeta(answer);
  return {
    id: `chatcmpl-${model}`,
    object: 'chat.completion',
    created: EPOCH,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: meta.finish }],
    usage: {
      prompt_tokens: meta.promptTokens,
      completion_tokens: meta.completionTokens,
      total_tokens: meta.promptTokens + meta.completionTokens,
    },
  } as Json;
}
