import { NextRequest, NextResponse } from 'next/server';

// Server-side HuggingFace inference proxy — keeps the API key out of the browser.
// POST /api/hf-inference
// Body: { model: string; inputs: unknown; parameters?: unknown }

export async function POST(req: NextRequest) {
  const apiKey = process.env.HUGGINGFACE_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ error: 'HUGGINGFACE_API_KEY not configured' }, { status: 500 });
  }

  let body: { model: string; inputs: unknown; parameters?: unknown };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  const { model, inputs, parameters } = body;
  if (!model || inputs === undefined) {
    return NextResponse.json({ error: 'model and inputs are required' }, { status: 400 });
  }

  const hfUrl = `https://api-inference.huggingface.co/models/${model}`;

  try {
    const hfRes = await fetch(hfUrl, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ inputs, ...(parameters ? { parameters } : {}) }),
    });

    const contentType = hfRes.headers.get('content-type') ?? '';

    if (!hfRes.ok) {
      const errText = await hfRes.text();
      return NextResponse.json(
        { error: `HuggingFace API error ${hfRes.status}`, detail: errText },
        { status: hfRes.status },
      );
    }

    // Binary response (image segmentation masks, feature embeddings as raw bytes, etc.)
    if (contentType.includes('application/octet-stream') || contentType.includes('image/')) {
      const buf = await hfRes.arrayBuffer();
      return new NextResponse(buf, {
        status: 200,
        headers: { 'Content-Type': contentType },
      });
    }

    const json = await hfRes.json();
    return NextResponse.json(json);
  } catch (err) {
    return NextResponse.json(
      { error: 'Upstream fetch failed', detail: String(err) },
      { status: 502 },
    );
  }
}
