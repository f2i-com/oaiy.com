/**
 * Cards kept clear of the cards that grew, and given their places back when those fold again.
 *
 * A card grows when a run fills it in (a picture in Image View, a result in Output) and when it is opened (its chevron,
 * or Expand for every card at once). A card laid out for how it was then lies over its neighbours: the cards below it
 * in its column (under where it ended, across its width as it was) move down together, by as much as it now reaches
 * into the nearest of them, plus a gap; and the cards beside it to its right (past where it ended, across its height
 * now) move right together the same way. What a moved card then reaches moves on, the same way.
 *
 * Every card that grew in one go is taken at once, each against where the others are by then: Expand on a flow of
 * several columns used to be taken card by card against where every card had been before any of them moved (and only
 * ever downwards), so cards went over one another and over the next column.
 *
 * Each move is remembered as a push by the card that grew (`Pushes`): when that card is folded again, the cards it
 * pushed go back by as much (`pullBack`), where that leaves them clear, so folding every card gives the flow back its
 * layout rather than the gaps the opening made. A card that is not where its pushes left it (moved by hand, laid out
 * again, an undo) is not pulled: its pushes are forgotten.
 */
export interface CardBox {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

/** A card's size before it grew. */
export interface SizeBefore {
  width: number;
  height: number;
}

/** What moved a card, and by how much. */
export interface Push {
  by: string;
  dx: number;
  dy: number;
}

/** The cards that were pushed: where their pushes left them, and the pushes. */
export type Pushes = Map<string, { x: number; y: number; pushes: Push[] }>;

export interface Respaced {
  /** The new position of each card to move, by id. */
  moves: Map<string, { x: number; y: number }>;
  /** The pushes made, by the card moved. */
  pushes: Map<string, Push[]>;
}

// (a card started level with another's foot, or its edge, counts as below or beside it)
const SLACK = 4;

const overlaps = (a: CardBox, b: CardBox) =>
  Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x) > 1 && Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y) > 1;
const acrossWidth = (c: CardBox, x: number, width: number) => c.x < x + width && c.x + c.width > x;
const acrossHeight = (c: CardBox, y: number, height: number) => c.y < y + height && c.y + c.height > y;

/**
 * The cards that grew (`before`: their sizes before, by id; `cards` at their sizes now, where they are now) and every
 * card they now reach, moved clear: down for the cards below one, right for the cards beside it.
 */
export function clearGrown(cards: readonly CardBox[], before: ReadonlyMap<string, SizeBefore>, gap = 24): Respaced {
  const now = new Map(cards.map((c) => [c.id, { ...c }]));
  const start = new Map(cards.map((c) => [c.id, { ...c, ...(before.get(c.id) ?? {}) }]));
  const pushes = new Map<string, Push[]>();
  const push = (c: CardBox, by: string, dx: number, dy: number) => {
    c.x += dx;
    c.y += dy;
    const list = pushes.get(c.id) ?? [];
    const same = list.find((p) => p.by === by);
    if (same) {
      same.dx += dx;
      same.dy += dy;
    } else list.push({ by, dx, dy });
    pushes.set(c.id, list);
  };
  // What a moved card reaches moves on the same way: below it, or beside it (cards that lay over each other already,
  // before any of this, are left as they were).
  const carryOn = (moved: CardBox[], by: string, down: boolean) => {
    const queue = [...moved];
    for (let guard = 0; queue.length && guard < cards.length * cards.length; guard++) {
      const m = queue.shift()!;
      const m0 = start.get(m.id)!;
      for (const o of now.values()) {
        if (o.id === m.id || !overlaps(m, o) || overlaps(m0, start.get(o.id)!)) continue;
        const o0 = start.get(o.id)!;
        if (down && o0.y >= m0.y) push(o, by, 0, m.y + m.height + gap - o.y);
        else if (!down && o0.x >= m0.x) push(o, by, m.x + m.width + gap - o.x, 0);
        else continue;
        queue.push(o);
      }
    }
  };
  const grown = cards
    .filter((c) => {
      const b = before.get(c.id);
      return b && (c.height > b.height + 0.5 || c.width > b.width + 0.5);
    })
    .sort((a, b) => a.y - b.y || a.x - b.x);
  for (const g0 of grown) {
    const g = now.get(g0.id)!;
    const was = before.get(g.id)!;
    if (g.height > was.height) {
      // (its column as it was: a card past its old right edge is beside it, and moves right if it is reached at all)
      const below = [...now.values()].filter((c) => c.id !== g.id && c.y >= g.y + was.height - SLACK && acrossWidth(c, g.x, was.width));
      const by = below.length ? g.y + g.height + gap - Math.min(...below.map((c) => c.y)) : 0;
      if (by > 0) {
        for (const c of below) push(c, g.id, 0, by);
        carryOn(below, g.id, true);
      }
    }
    if (g.width > was.width) {
      const beside = [...now.values()].filter((c) => c.id !== g.id && c.x >= g.x + was.width - SLACK && acrossHeight(c, g.y, g.height));
      const by = beside.length ? g.x + g.width + gap - Math.min(...beside.map((c) => c.x)) : 0;
      if (by > 0) {
        for (const c of beside) push(c, g.id, by, 0);
        carryOn(beside, g.id, false);
      }
    }
  }
  const moves = new Map<string, { x: number; y: number }>();
  for (const id of pushes.keys()) {
    const c = now.get(id)!;
    moves.set(id, { x: c.x, y: c.y });
  }
  return { moves, pushes };
}

/** The pushes of a respace added to those remembered (`known`, changed in place), each card at where it now is. */
export function remember(known: Pushes, done: Respaced): void {
  for (const [id, list] of done.pushes) {
    const at = done.moves.get(id)!;
    // (pushes from before it was moved some other way are forgotten: they no longer say where it was)
    const from = { x: at.x - list.reduce((n, p) => n + p.dx, 0), y: at.y - list.reduce((n, p) => n + p.dy, 0) };
    const rec = known.get(id);
    const had = rec && Math.abs(rec.x - from.x) <= 1 && Math.abs(rec.y - from.y) <= 1 ? rec.pushes : [];
    const merged = had.map((p) => ({ ...p }));
    for (const p of list) {
      const same = merged.find((q) => q.by === p.by);
      if (same) {
        same.dx += p.dx;
        same.dy += p.dy;
      } else merged.push({ ...p });
    }
    known.set(id, { x: at.x, y: at.y, pushes: merged });
  }
}

/**
 * The cards pushed by the ones that folded (`folded`), back by as much as they were pushed, where that leaves each
 * clear of every other card; `known` is changed in place (a card pulled back keeps only its other pushes, a card that
 * is not where its pushes left it loses them all).
 */
export function pullBack(cards: readonly CardBox[], folded: ReadonlySet<string>, known: Pushes): Map<string, { x: number; y: number }> {
  const now = new Map(cards.map((c) => [c.id, { ...c }]));
  const moves = new Map<string, { x: number; y: number }>();
  const waiting: string[] = [];
  for (const [id, rec] of known) {
    const c = now.get(id);
    if (!c || Math.abs(c.x - rec.x) > 1 || Math.abs(c.y - rec.y) > 1) {
      known.delete(id);
      continue;
    }
    if (rec.pushes.some((p) => folded.has(p.by))) waiting.push(id);
  }
  // (in passes, so a card waits for the one above it or before it to go back first)
  for (let changed = true; changed && waiting.length; ) {
    changed = false;
    for (const id of [...waiting]) {
      const rec = known.get(id)!;
      const c = now.get(id)!;
      const back = rec.pushes.filter((p) => folded.has(p.by));
      const to = { ...c, x: c.x - back.reduce((s, p) => s + p.dx, 0), y: c.y - back.reduce((s, p) => s + p.dy, 0) };
      if ([...now.values()].some((o) => o.id !== id && overlaps(to, o))) continue;
      c.x = to.x;
      c.y = to.y;
      moves.set(id, { x: to.x, y: to.y });
      const rest = rec.pushes.filter((p) => !folded.has(p.by));
      if (rest.length) known.set(id, { x: to.x, y: to.y, pushes: rest });
      else known.delete(id);
      waiting.splice(waiting.indexOf(id), 1);
      changed = true;
    }
  }
  return moves;
}

/** One card that grew: the new `y` of each card below it to move, by id (empty where nothing below is reached). */
export function clearBelow(cards: readonly CardBox[], grownId: string, heightBefore: number, gap = 24): Map<string, number> {
  const grown = cards.find((c) => c.id === grownId);
  const out = new Map<string, number>();
  if (!grown) return out;
  const { moves } = clearGrown(cards, new Map([[grownId, { width: grown.width, height: heightBefore }]]), gap);
  for (const [id, at] of moves) out.set(id, at.y);
  return out;
}

/** A card as the canvas has it: where it is, its size when last measured, and whether it is folded. */
export interface CardNode {
  id: string;
  parentId?: string;
  position: { x: number; y: number };
  measured?: { width?: number; height?: number };
  folded: boolean;
}

/**
 * What the canvas does with each measurement of its cards (OAIYBuilder's onNodesChange): the moves that keep them clear
 * of one another, when they grew while a run fills them in (`running`), or when they were opened (one by its chevron
 * or all at once) and for a moment after (a code editor in one loads, and it grows again); and, when they were folded
 * again, their room given back. Nothing for a card measured as the flow opens, nor for one that changes size by itself
 * at another time. Every card measured in one go is taken at once.
 */
export class Respacer {
  /** Each card's folded state when it was last measured: a change is an opening or a fold. */
  private readonly folded = new Map<string, boolean>();
  /** A card just opened, until when its growth is still kept clear. */
  private readonly openedUntil = new Map<string, number>();
  /** Who pushed which card, and where that left it. */
  readonly pushes: Pushes = new Map();
  /**
   * Moves made, until the canvas has them (a second measurement can come before they are drawn): where each card was
   * sent, and from where. Only while the card is still where it was sent from: one moved since (by hand) is where it is.
   */
  private readonly pending = new Map<string, { x: number; y: number; from: { x: number; y: number }; t: number }>();

  constructor(private readonly openedFor = 2000, private readonly gap = 24) {}

  measured(nodes: readonly CardNode[], sizes: ReadonlyMap<string, { width: number; height: number }>, now: number, running: boolean): Map<string, { x: number; y: number }> {
    const moves = new Map<string, { x: number; y: number }>();
    if (!sizes.size) return moves;
    const grew = new Map<string, SizeBefore>();
    const folded = new Set<string>();
    for (const n of nodes) {
      const size = sizes.get(n.id);
      if (!size) continue;
      const was = this.folded.get(n.id);
      this.folded.set(n.id, n.folded);
      const before = n.measured;
      if (!before?.width || !before.height) continue;
      const turned = was !== undefined && was !== n.folded;
      if (turned && n.folded) {
        this.openedUntil.delete(n.id);
        if (size.height < before.height - 4 || size.width < before.width - 4) folded.add(n.id);
        continue;
      }
      if (turned) this.openedUntil.set(n.id, now + this.openedFor);
      if (!running && !turned && (this.openedUntil.get(n.id) ?? 0) < now) continue;
      if (size.height > before.height + 4 || size.width > before.width + 4) grew.set(n.id, { width: before.width, height: before.height });
    }
    if (!grew.size && !folded.size) return moves;
    for (const n of nodes) {
      const p = this.pending.get(n.id);
      if (p && (Math.abs(p.from.x - n.position.x) > 0.5 || Math.abs(p.from.y - n.position.y) > 0.5 || now - p.t > 1000)) this.pending.delete(n.id);
    }
    let cards: CardBox[] = nodes
      .filter((n) => !n.parentId)
      .map((n) => {
        const at = this.pending.get(n.id) ?? n.position;
        const size = sizes.get(n.id);
        return { id: n.id, x: at.x, y: at.y, width: size?.width ?? n.measured?.width ?? 0, height: size?.height ?? n.measured?.height ?? 0 };
      });
    const take = (made: Map<string, { x: number; y: number }>) => {
      for (const [id, at] of made) moves.set(id, at);
      cards = cards.map((c) => (made.has(c.id) ? { ...c, ...made.get(c.id)! } : c));
    };
    if (folded.size) take(pullBack(cards, folded, this.pushes));
    if (grew.size) {
      const done = clearGrown(cards, grew, this.gap);
      remember(this.pushes, done);
      take(done.moves);
    }
    for (const [id, at] of moves) {
      const n = nodes.find((c) => c.id === id);
      if (n) this.pending.set(id, { ...at, from: { ...n.position }, t: now });
    }
    return moves;
  }
}
