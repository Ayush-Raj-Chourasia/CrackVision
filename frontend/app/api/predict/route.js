import { NextResponse } from 'next/server';

export const runtime = 'nodejs';

export async function POST(request) {
  const backendUrl = process.env.CRACKVISION_API_URL;
  if (!backendUrl) {
    return NextResponse.json(
      { detail: 'Set CRACKVISION_API_URL in the Vercel environment.' },
      { status: 500 },
    );
  }

  const incoming = await request.formData();
  const image = incoming.get('image');
  const mode = incoming.get('mode') || 'accurate';
  const confidence = incoming.get('confidence') || '0.50';

  if (!image) {
    return NextResponse.json({ detail: 'Missing image upload.' }, { status: 400 });
  }

  const formData = new FormData();
  formData.append('mode', mode);
  formData.append('confidence', confidence);
  formData.append('image', image, image.name || 'upload.jpg');

  const response = await fetch(`${backendUrl.replace(/\/$/, '')}/predict`, {
    method: 'POST',
    body: formData,
  });

  const payload = await response.json();
  return NextResponse.json(payload, { status: response.status });
}
