import {
  DEFAULT_MAX_MEMORY_ENTRIES,
  DEFAULT_MAX_VALUE_SIZE_BYTES,
  CIRCULAR_REF_SIZE_ESTIMATE,
  DEFAULT_VALUE_SIZE_ESTIMATE,
} from '../constants';

/**
 * BoundedMap - A Map with size limits and LRU eviction.
 * Prevents unbounded memory growth in agent memory.
 *
 * Features:
 * - Maximum entry limit with LRU eviction
 * - Maximum value size limit to prevent memory bloat
 * - Standard Map interface for easy adoption
 *
 * @example
 * ```typescript
 * const memory = new BoundedMap<string, object>({
 *   maxEntries: 1000,      // Max 1000 entries
 *   maxValueSize: 1024 * 1024, // Max 1MB per value
 * });
 *
 * memory.set('key', { data: 'value' });
 * const value = memory.get('key'); // Also refreshes LRU position
 * ```
 */
export class BoundedMap<K, V> {
  private map: Map<K, V> = new Map();
  private readonly maxEntries: number;
  private readonly maxValueSize: number;

  constructor(options?: { maxEntries?: number; maxValueSize?: number }) {
    this.maxEntries = options?.maxEntries ?? DEFAULT_MAX_MEMORY_ENTRIES;
    this.maxValueSize = options?.maxValueSize ?? DEFAULT_MAX_VALUE_SIZE_BYTES;
  }

  private estimateSize(value: V): number {
    if (value === null || value === undefined) return 0;
    if (typeof value === 'string') return value.length * 2; // UTF-16
    if (typeof value === 'number') return 8;
    if (typeof value === 'boolean') return 4;
    if (typeof value === 'object') {
      try {
        return JSON.stringify(value).length * 2;
      } catch {
        return CIRCULAR_REF_SIZE_ESTIMATE;
      }
    }
    return DEFAULT_VALUE_SIZE_ESTIMATE;
  }

  get(key: K): V | undefined {
    const value = this.map.get(key);
    if (value !== undefined) {
      // Move to end for LRU (delete and re-add)
      this.map.delete(key);
      this.map.set(key, value);
    }
    return value;
  }

  set(key: K, value: V): this {
    // Check value size
    const valueSize = this.estimateSize(value);
    if (valueSize > this.maxValueSize) {
      console.warn(`[BoundedMap] Value too large (${valueSize} bytes), max is ${this.maxValueSize}`);
      return this;
    }

    // If key exists, delete it first (for LRU ordering)
    if (this.map.has(key)) {
      this.map.delete(key);
    }

    // Evict oldest entries if at capacity
    while (this.map.size >= this.maxEntries) {
      const oldestKey = this.map.keys().next().value;
      if (oldestKey !== undefined) {
        this.map.delete(oldestKey);
      } else {
        break;
      }
    }

    this.map.set(key, value);
    return this;
  }

  delete(key: K): boolean {
    return this.map.delete(key);
  }

  has(key: K): boolean {
    return this.map.has(key);
  }

  clear(): void {
    this.map.clear();
  }

  get size(): number {
    return this.map.size;
  }

  entries(): IterableIterator<[K, V]> {
    return this.map.entries();
  }

  keys(): IterableIterator<K> {
    return this.map.keys();
  }

  values(): IterableIterator<V> {
    return this.map.values();
  }

  forEach(callback: (value: V, key: K, map: Map<K, V>) => void): void {
    this.map.forEach(callback);
  }

  [Symbol.iterator](): IterableIterator<[K, V]> {
    return this.map[Symbol.iterator]();
  }
}
