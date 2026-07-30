/**
 * Metrics Module - Provides observability for workflow execution.
 *
 * Tracks:
 * - Compilation times
 * - Workflow execution times
 * - Node execution counts
 * - Queue statistics
 *
 * Usage:
 * ```typescript
 * import { metrics } from './metrics';
 *
 * // Record a timing
 * const end = metrics.startTimer('compilation');
 * // ... do work ...
 * end(); // Records the duration
 *
 * // Get summary
 * console.log(metrics.getSummary());
 * ```
 */

import { createLogger } from './logger.js';

const logger = createLogger('Metrics');

export interface TimingStats {
  count: number;
  totalMs: number;
  minMs: number;
  maxMs: number;
  avgMs: number;
}

export interface MetricsSummary {
  timings: Record<string, TimingStats>;
  counters: Record<string, number>;
  gauges: Record<string, number>;
}

/**
 * Metrics collector for workflow observability
 */
class MetricsCollector {
  private timings: Map<string, number[]> = new Map();
  private counters: Map<string, number> = new Map();
  private gauges: Map<string, number> = new Map();
  private enabled: boolean = true;

  /**
   * Enable or disable metrics collection.
   * Disabling prevents performance overhead in production if not needed.
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
  }

  /**
   * Check if metrics collection is enabled
   */
  isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * Start a timer and return a function to stop it.
   * The duration is recorded when the returned function is called.
   *
   * @param name - Name of the timing metric
   * @returns Function to call when the operation completes
   */
  startTimer(name: string): () => number {
    if (!this.enabled) {
      return () => 0;
    }

    const start = performance.now();
    return () => {
      const duration = performance.now() - start;
      this.recordTiming(name, duration);
      return duration;
    };
  }

  /**
   * Record a timing value directly
   */
  recordTiming(name: string, durationMs: number): void {
    if (!this.enabled) return;

    const values = this.timings.get(name) || [];
    values.push(durationMs);
    // Bound the per-metric history (ring buffer). `metrics` is a process-lifetime
    // singleton whose reset() is never called in production, and 'compilation'
    // grows per subflow compile — so an unbounded array is a slow real leak.
    if (values.length > 1000) values.shift();
    this.timings.set(name, values);
  }

  /**
   * Increment a counter
   */
  increment(name: string, value: number = 1): void {
    if (!this.enabled) return;

    const current = this.counters.get(name) || 0;
    this.counters.set(name, current + value);
  }

  /**
   * Set a gauge value (current state, not cumulative)
   */
  setGauge(name: string, value: number): void {
    if (!this.enabled) return;

    this.gauges.set(name, value);
  }

  /**
   * Get the current value of a gauge
   */
  getGauge(name: string): number {
    return this.gauges.get(name) || 0;
  }

  /**
   * Get statistics for a timing metric
   */
  getTimingStats(name: string): TimingStats | null {
    const values = this.timings.get(name);
    if (!values || values.length === 0) return null;

    const totalMs = values.reduce((a, b) => a + b, 0);
    return {
      count: values.length,
      totalMs,
      // reduce, not Math.min(...values) — a large samples array would blow the
      // call-stack arg limit (RangeError) on the spread.
      minMs: values.reduce((m, v) => (v < m ? v : m), values[0]),
      maxMs: values.reduce((m, v) => (v > m ? v : m), values[0]),
      avgMs: totalMs / values.length,
    };
  }

  /**
   * Get a summary of all collected metrics
   */
  getSummary(): MetricsSummary {
    const timings: Record<string, TimingStats> = {};
    for (const [name] of this.timings) {
      const stats = this.getTimingStats(name);
      if (stats) {
        timings[name] = stats;
      }
    }

    return {
      timings,
      counters: Object.fromEntries(this.counters),
      gauges: Object.fromEntries(this.gauges),
    };
  }

  /**
   * Log a summary of metrics at info level
   */
  logSummary(): void {
    const summary = this.getSummary();

    // Log timing stats
    for (const [name, stats] of Object.entries(summary.timings)) {
      logger.info(`Timing: ${name}`, {
        count: stats.count,
        avgMs: Math.round(stats.avgMs * 100) / 100,
        minMs: Math.round(stats.minMs * 100) / 100,
        maxMs: Math.round(stats.maxMs * 100) / 100,
      });
    }

    // Log counters
    for (const [name, value] of Object.entries(summary.counters)) {
      logger.info(`Counter: ${name} = ${value}`);
    }

    // Log gauges
    for (const [name, value] of Object.entries(summary.gauges)) {
      logger.info(`Gauge: ${name} = ${value}`);
    }
  }

  /**
   * Reset all metrics
   */
  reset(): void {
    this.timings.clear();
    this.counters.clear();
    this.gauges.clear();
  }
}

// Export a singleton instance
export const metrics = new MetricsCollector();

// Export the class for testing or custom instances
export { MetricsCollector };
