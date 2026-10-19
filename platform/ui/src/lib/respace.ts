/**
 * Cards kept clear of one that grew. A run fills cards in (a picture in Image View, a result in Output), and a card
 * laid out for how it was then lies over the one below it. The cards below it in its column (under where it ended,
 * and across its width) move down together, by as much as it now reaches into the nearest of them, plus a gap.
 */
export interface CardBox {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

/** The new `y` of each card to move, by id; empty where nothing below is reached. */
export function clearBelow(cards: readonly CardBox[], grownId: string, heightBefore: number, gap = 24): Map<string, number> {
  const moves = new Map<string, number>();
  const grown = cards.find((c) => c.id === grownId);
  if (!grown || !(grown.height > heightBefore)) return moves;
  const endedBefore = grown.y + heightBefore;
  const endsNow = grown.y + grown.height;
  // (below where it ended, a little allowed for a card started level with its foot; across its width)
  const below = cards.filter((c) => c.id !== grownId && c.y >= endedBefore - 4 && c.x < grown.x + grown.width && c.x + c.width > grown.x);
  if (!below.length) return moves;
  const nearest = Math.min(...below.map((c) => c.y));
  const by = endsNow + gap - nearest;
  if (by <= 0) return moves;
  for (const c of below) moves.set(c.id, c.y + by);
  return moves;
}
