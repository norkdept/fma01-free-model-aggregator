/**
 * How long a failing provider is left alone, and how long one is waited on.
 *
 * The ranking in router.ts already remembers that a provider refused — its Health counts refusals and
 * successes and moves a refusing provider back in the order. Counting is not timing, though: a provider
 * answering 429 with `retry-after` was tried again on the very next request because nothing remembered
 * *when* it was allowed back, and a provider that accepted the connection and then stalled held the
 * request open with no deadline at all, so the failover chain never reached the providers behind it.
 * Both faults are about the clock, and the clock lives here.
 */

/** What a provider said about waiting, before it is turned into a deadline. */
export interface RefusalAdvice {
  /** Milliseconds the provider asked to be left alone, from `retry-after`, or null when it said nothing. */
  retryAfterMs: number | null;
  /** True when the refusal is about a rate limit rather than a fault on the provider's side. */
  rateLimited: boolean;
}

/** The quiet window each provider has asked for, in wall-clock milliseconds. */
export type QuietWindows = Map<string, number>;

/** The longest wait worth honouring. */
export const MAX_QUIET_MS = 15 * 60 * 1000;

/** The deadline a request may spend on one provider before it is treated as refusing. */
export const CANDIDATE_TIMEOUT_MS = 60 * 1000;

/** The delay applied to a rate limit that named no duration; doubled per consecutive refusal. */
export const BASE_BACKOFF_MS = 1000;

/** No provider is quiet, which is what a trial starts with. */
export function freshQuiet(): QuietWindows {
  return new Map();
}

/**
 * Read the advice a refusal carries.
 *
 * `retry-after` is specified as either delta-seconds or an HTTP date, and free tiers send both, so the
 * two forms are told apart before either is read: a bare `parseInt` would read the date form as a number
 * of seconds that is really the year. `now` is supplied by the caller for the date form, so a replayed
 * trial reads the header against the trial's own clock and not the replaying process's.
 */
export function adviceFromRefusal(response: Response, now: number): RefusalAdvice {
  const header = response.headers.get('retry-after');
  const rateLimited = response.status === 429 || response.status === 402;
  return { retryAfterMs: parseRetryAfter(header, now), rateLimited };
}

/** Turn a `retry-after` header value into milliseconds, or null when it is absent or unreadable. */
export function parseRetryAfter(header: string | null, now: number): number | null {
  if (header === null) return null;
  const trimmed = header.trim();
  if (trimmed === '') return null;
  // Delta-seconds is digits only. `Number` would also accept "1e3", "0x10" and "-5", none of which a
  // provider means by this header, so the shape is checked before the value is read.
  if (/^\d+$/.test(trimmed)) {
    const seconds = Number(trimmed);
    if (!Number.isFinite(seconds)) return null;
    return Math.min(seconds * 1000, MAX_QUIET_MS);
  }
  // The HTTP-date form, which a few providers send instead. An unparseable date is no advice at all.
  const at = Date.parse(trimmed);
  if (Number.isNaN(at)) return null;
  return Math.min(Math.max(0, at - now), MAX_QUIET_MS);
}

/**
 * How long to leave a provider alone after a refusal.
 *
 * A provider that named a duration is believed, clamped so a `retry-after: 86400` ("come back tomorrow")
 * does not read as "this provider is the only choice" — it is ranked last meanwhile, and the others are
 * still tried. A rate limit that named nothing gets an exponential backoff from its consecutive-refusal
 * count. Any other refusal is not about time and sets no window: the next request should try it.
 */
export function quietFor(advice: RefusalAdvice, consecutiveRefusals: number): number {
  if (advice.retryAfterMs !== null) return Math.min(advice.retryAfterMs, MAX_QUIET_MS);
  if (!advice.rateLimited) return 0;
  const grown = BASE_BACKOFF_MS * Math.pow(2, Math.max(0, consecutiveRefusals - 1));
  return Math.min(grown, MAX_QUIET_MS);
}

/**
 * Record the window a refusal asks for.
 *
 * A provider already quiet for longer keeps that deadline: one short refusal must not shorten a long
 * wait that was already promised, or a provider could be kept in rotation by failing briefly.
 */
export function noteQuiet(windows: QuietWindows, providerId: string, advice: RefusalAdvice, consecutiveRefusals: number, now: number): void {
  const quiet = quietFor(advice, consecutiveRefusals);
  if (quiet <= 0) return;
  const until = now + quiet;
  const current = windows.get(providerId) ?? 0;
  if (until > current) windows.set(providerId, until);
}

/** True when a provider is still inside the window a refusal asked for. */
export function isQuiet(windows: QuietWindows, providerId: string, now: number): boolean {
  return (windows.get(providerId) ?? 0) > now;
}

/**
 * The providers that may be tried now, in ranking order, with the quiet ones pushed to the back.
 *
 * The quiet providers are kept rather than dropped: if every provider is inside a window, refusing the
 * request outright would fail it, when trying one a little early is the better answer. The order is
 * preserved within each group, so the ranking still decides among equals.
 */
export function partitionQuiet<T extends { probe: { provider: { id: string } } }>(candidates: T[], windows: QuietWindows, now: number): T[] {
  const ready: T[] = [];
  const resting: T[] = [];
  for (const candidate of candidates) {
    if (isQuiet(windows, candidate.probe.provider.id, now)) resting.push(candidate);
    else ready.push(candidate);
  }
  return ready.concat(resting);
}

/**
 * Run one provider call under a deadline.
 *
 * A free tier that accepts the connection and then stops writing would otherwise hold the request open
 * until the client gave up, and with it the whole chain: the aggregator would never reach the providers
 * behind it. The signal aborts the fetch, and the caller treats the abort as a refusal so the walk
 * continues. The timer is always cleared, so a fast call leaves nothing pending.
 */
export async function withDeadline<T>(work: (signal: AbortSignal) => Promise<T>, timeoutMs = CANDIDATE_TIMEOUT_MS): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await work(controller.signal);
  } finally {
    clearTimeout(timer);
  }
}
