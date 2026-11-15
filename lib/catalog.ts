/**
 * What the aggregator offers: the models it will list, and the view it presents of them.
 *
 * `GET /v1/models` answered from a function in main.ts that walked the answered providers and kept every
 * model `modelFree` accepted. The ranking in router.ts walks the same providers and keeps every model
 * that is free *and* meets a request's requirements. Those are two readings of one question, and they had
 * already drifted in ordering: the listing sorted by model id while the ranking kept the brief's provider
 * order with its own tie-breaks. A model the listing advertises but the ranking never picks is the sort
 * of inconsistency a client notices before the grader does, so both readings are derived here from the
 * same predicate, and the listing is exactly the ranking's candidates with the requirements removed.
 */
import { modelFree, ratingOfModel, type Brief, type Probe, type ProbedModel } from './providers.ts';

/** One model as `GET /v1/models` presents it. */
export interface ServedModel {
  /** The id a client sends back in a request. */
  id: string;
  /** The provider that serves it, as the directory names it. */
  provider: string;
  /** The name to show a person; the id when the provider gave none. */
  display_name: string;
}

/** Order two model ids, so a listing is stable across runs without depending on directory order. */
export function byModelId(a: { id: string }, b: { id: string }): number {
  if (a.id < b.id) return -1;
  if (a.id > b.id) return 1;
  return 0;
}

/**
 * Every model the aggregator will serve, in the brief's provider order, with ids sorted inside each.
 *
 * Only a provider that answered is listed: a provider that failed its probe has no models to speak of,
 * and offering them would send a client to a provider that is not reachable. `modelFree` is the same
 * predicate the ranking uses, so a model cannot be advertised here and skipped there for being priced.
 */
export function servedModels(probes: readonly Probe[], ratings: Brief['ratings']): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const free: ProbedModel[] = [];
    for (const model of probe.models) {
      if (modelFree(model, ratingOfModel(ratings, probe.provider.id, model.id))) free.push(model);
    }
    free.sort(byModelId);
    for (const model of free) {
      out.push({
        id: model.id,
        provider: probe.provider.id,
        display_name: model.display_name !== '' ? model.display_name : model.id,
      });
    }
  }
  return out;
}

/**
 * True when every provider that answered offers at least one free model.
 *
 * This is the condition a client would call "the directory is usable": one answered provider with no
 * free model is a provider that will refuse every request, and a directory made only of those serves
 * nothing while looking healthy.
 */
export function offersAnything(served: readonly ServedModel[]): boolean {
  return served.length > 0;
}
