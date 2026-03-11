/**
 * Next.js API Route — Proxy to backend /api/ingest
 * Handles multipart file upload forwarding.
 */

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function POST(request) {
    try {
        const formData = await request.formData();

        const res = await fetch(`${BACKEND_URL}/api/ingest`, {
            method: 'POST',
            body: formData,
        });

        const data = await res.json();
        return Response.json(data, { status: res.status });
    } catch (err) {
        return Response.json(
            { detail: `Backend unreachable: ${err.message}` },
            { status: 502 }
        );
    }
}
