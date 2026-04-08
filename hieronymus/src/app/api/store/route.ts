import { NextRequest, NextResponse } from 'next/server';

// In-memory store — resets on serverless cold start
// In production this becomes Vercel KV. The API shape stays the same.
const store = new Map<string, {
  S_k: number;
  S_t: number;
  S_e: number;
  name: string;
  domain: string;
  metadata?: Record<string, unknown>;
  created: number;
}>();

// Also populate the match route's store when items are added
// Since they run in the same process during dev, we share state via a global
function getGlobalStore() {
  const g = globalThis as Record<string, unknown>;
  if (!g.__hieronymus_store) {
    g.__hieronymus_store = store;
  }
  return g.__hieronymus_store as typeof store;
}

export async function POST(req: NextRequest) {
  try {
    const { S_k, S_t, S_e, name, domain, metadata } = await req.json();

    if (typeof S_k !== 'number' || typeof S_t !== 'number' || typeof S_e !== 'number') {
      return NextResponse.json(
        { error: 'S_k, S_t, and S_e are required as numbers' },
        { status: 400 }
      );
    }

    const id = crypto.randomUUID();
    const entry = {
      S_k,
      S_t,
      S_e,
      name: name || 'Untitled',
      domain: domain || 'unknown',
      metadata: metadata || {},
      created: Date.now(),
    };

    const globalStore = getGlobalStore();
    globalStore.set(id, entry);

    return NextResponse.json({ id, stored: true });
  } catch {
    return NextResponse.json({ error: 'Invalid request body' }, { status: 400 });
  }
}

export async function GET() {
  const globalStore = getGlobalStore();
  const items = Array.from(globalStore.entries()).map(([id, v]) => ({ id, ...v }));
  return NextResponse.json(items);
}
