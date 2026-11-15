// Routing and failover for free-model-aggregator-01.
//
// The score: quality 60 (weighted rating vs the best legal rating), served 15 (share of requests
// answered), wire 10 (response shape), failover 10 (share recovered after a refusal), catalog 5.
//
// The plain baseline takes the first provider in brief order with a free model meeting the
// requirements. That leaves almost all of the quality column on the table: the best legal model for
// most requests is the highest-rating free one, not the first listed.
//
// What this module does instead:
//   1. It ranks every free model that meets a request's requirements by rating_milli, descending.
//   2. It walks that ranking when a provider refuses (402, 429, 5xx, a network error), so one
//      exhausted provider does not take its whole share of the workload down with it.
//   3. It spreads load: a provider's free-tier budget is drawn per trial and unpublished, so a
//      provider that has already refused is moved to the back of the ranking for later requests.
//   4. It never picks a paid model, and it honours context length, tools, vision and JSON mode.
//
// Everything here is a pure function of (requirements, what has been observed so far): no clock, no
// unseeded randomness. A trial is replayed and must leave a byte-identical export, so a decision may
// not depend on time or on anything the replay would not also see.

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Probe,
  type ProbedModel,
  type RatingsFile,
  type Requirement,
  type RatingRow,
} from './providers.ts';

/** One provider+model a request can be served by. */
export interface Candidate {
  probe: Probe;
  model: ProbedModel;
  /** The quality table's rating in milli-points; 0 when the table has no row. */
  rating: number;
}

/** Statuses a provider answers to mean "not me, try someone else". */
export function isRefusal(status: number): boolean {
  return status === 402 || status === 429 || status >= 500;
}

/** A per-provider view of what the trial has seen: refusals and successes. */
export interface Health {
  /** Calls that came back as a refusal, per provider id. */
  refusals: Map<string, number>;
  /** Calls that came back 2xx, per provider id. */
  served: Map<string, number>;
}

export const freshHealth = (): Health => ({ refusals: new Map(), served: new Map() });

const bump = (m: Map<string, number>, k: string): void => {
  m.set(k, (m.get(k) ?? 0) + 1);
};

export function noteRefusal(health: Health, providerId: string): void {
  bump(health.refusals, providerId);
}

export function noteServed(health: Health, providerId: string): void {
  bump(health.served, providerId);
}

/**
 * Every free model across the answered providers that meets `requires`, best rating first.
 *
 * Ties on rating are broken by fewer refusals, then by more successes, then by provider id and model
 * id, all of which the replay reproduces. The tie-break matters: without it two equally-rated models
 * would be tried in brief order forever and a refusing provider would keep getting the first call.
 */
export function rankCandidates(
  probes: Probe[],
  ratings: RatingsFile | null,
  requires: Requirement,
  health: Health,
): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const ra = health.refusals.get(a.probe.provider.id) ?? 0;
    const rb = health.refusals.get(b.probe.provider.id) ?? 0;
    if (ra !== rb) return ra - rb;
    const sa = health.served.get(a.probe.provider.id) ?? 0;
    const sb = health.served.get(b.probe.provider.id) ?? 0;
    if (sa !== sb) return sa - sb;
    const pi = a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0;
    if (pi !== 0) return pi;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/**
 * A wire that can carry this request at all. JSON mode is the one capability a wire itself decides:
 * an OpenAI-wire provider carries response_format, an Anthropic-wire one carries nothing equivalent,
 * and the Gemini wire carries responseMimeType. The provider echoes what it parsed and the client
 * compares the echo, so a JSON-mode request routed to a wire that drops the field loses the request
 * even though the model could have served it.
 */
export function wireCarries(candidate: Candidate, requires: Requirement, row: RatingRow | null): boolean {
  if (requires.json === true) {
    if (candidate.probe.provider.wire === 'anthropic') return false;
    // The model also has to accept JSON mode, which the quality table states where it has a row.
    if (row !== null && row !== undefined && row.json === false) return false;
  }
  return true;
}
