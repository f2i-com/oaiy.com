/**
 * Browser extraction helpers — page-context scripts evaluated by
 * Browser.evaluateJson.
 *
 * Each helper exports a script string (or factory for one) and the
 * shape of the value the script returns. plugin-scraper composes these
 * to enrich a website without re-implementing extraction logic.
 */

export * from "./emails";
export * from "./phones";
export * from "./text";
export * from "./links";
export * from "./images";
export * from "./colors";
export * from "./logo";
export * from "./metadata";
export * from "./jsonld";
