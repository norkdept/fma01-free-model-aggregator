/**
 * Why a provider said no, and what the walk should do about it.
 *
 * The classifier in router.ts answers one question — "is this status a refusal?" — with a boolean, and
 * three unrelated answers go into it alike: a 402 means the free tier's billing hook fired and this
 * provider will keep saying no to this model for as long as the account has no credit; a 429 means the
 * quota for this moment is spent and it will answer again later; a 5xx means the provider is having a
 * bad time and will most likely answer again later too. Failover behaves the same for all three — try
 * the next provider — but the *record* must not: only one of them is about time, and treating them
 * alike either retries a billing wall forever or forgets a quota that comes back.
 *
 * The distinction also decides whether the response body still matters. A refusal carries the reason a
 * client is owed when the walk ends with nothing served: the last 429's `retry-after` is worth more to
 * that client than a synthesised 502. So a refusal's body is read and kept, not drained and discarded.
 */

/** What a provider's status means for the walk. */
export type Verdict =
  /** Served it: 2xx. */
  | 'served'
  /** Out of quota for now: 429. Ranked back, quiet for as long as it asked, answers again later. */
  | 'rate_limited'
  /** Out of credit: 402. Ranked back for the whole trial; it will not get credit mid-run. */
  | 'unpaid'
  /** The provider is down: 5xx. Ranked back, most likely answers again later. */
  | 'outage'
  /** The request itself is wrong: a 4xx that is none of the above. Another provider refuses alike. */
  | 'rejected'
  /** Not the provider's fault or the request's: a thrown fetch, a deadline, a broken connection. */
  | 'unreachable';

/** Classify a status, or the absence of one when the call threw. */
export function verdictOf(status: number | null): Verdict {
  if (status === null) return 'unreachable';
  if (status >= 200 && status < 300) return 'served';
  if (status === 402) return 'unpaid';
  if (status === 429) return 'rate_limited';
  if (status >= 500) return 'outage';
  return 'rejected';
}

/** True when the walk should try the next provider rather than answering with this response. */
export function shouldFailover(verdict: Verdict): boolean {
  return verdict === 'rate_limited' || verdict === 'unpaid' || verdict === 'outage' || verdict === 'unreachable';
}

/**
 * True when the provider will answer again without anything changing.
 *
 * A rejection will not: sending the same request to another provider earns the same rejection, so the
 * walk stops and the client is told. This is the one case where the ranking is not consulted, and it is
 * why `shouldFailover` is not simply "not served".
 */
export function isPermanent(verdict: Verdict): boolean {
  return verdict === 'rejected' || verdict === 'unpaid';
}

/**
 * The status to answer with when the walk ran out of providers.
 *
 * A refusal is forwarded as it was, because it carries advice a client can act on: a 429 with
 * `retry-after` tells it exactly when to come back, and inventing a 502 instead throws that away. When
 * nothing was reached at all there is no provider status to forward, and 502 is the honest answer: the
 * aggregator could not reach the providers that were ranked for this request, which is not the client's
 * fault and not the aggregator's own 500.
 */
export function statusWhenExhausted(lastStatus: number | null): number {
  if (lastStatus === null) return 502;
  return lastStatus;
}

/** The type field a client expects on an error body, by wire. */
export function errorTypeFor(verdict: Verdict): string {
  if (verdict === 'rate_limited') return 'rate_limit_error';
  if (verdict === 'unpaid') return 'billing_error';
  if (verdict === 'rejected') return 'invalid_request_error';
  return 'upstream_error';
}
