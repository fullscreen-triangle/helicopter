import { NextRequest, NextResponse } from 'next/server';

// Access the shared global store (populated by /api/store)
function getGlobalStore() {
  const g = globalThis as Record<string, unknown>;
  if (!g.__hieronymus_store) {
    g.__hieronymus_store = new Map<string, { S_k: number; S_t: number; S_e: number; name: string; domain: string }>();
  }
  return g.__hieronymus_store as Map<string, { S_k: number; S_t: number; S_e: number; name: string; domain: string }>;
}

export async function POST(req: NextRequest) {
  try {
    const { S_k, S_t, S_e, topK = 10 } = await req.json();

    if (typeof S_k !== 'number' || typeof S_t !== 'number' || typeof S_e !== 'number') {
      return NextResponse.json(
        { error: 'S_k, S_t, and S_e are required as numbers' },
        { status: 400 }
      );
    }

    const store = getGlobalStore();

    // Compute S-distance to all items in store
    const results = Array.from(store.entries()).map(([id, item]) => {
      const d = Math.sqrt(
        (S_k - item.S_k) ** 2 + (S_t - item.S_t) ** 2 + (S_e - item.S_e) ** 2
      );
      return { id, name: item.name, domain: item.domain, distance: d, S_k: item.S_k, S_t: item.S_t, S_e: item.S_e };
    });

    // Sort by distance ascending, return top K
    results.sort((a, b) => a.distance - b.distance);

    return NextResponse.json(results.slice(0, topK));
  } catch {
    return NextResponse.json({ error: 'Invalid request body' }, { status: 400 });
  }
}
