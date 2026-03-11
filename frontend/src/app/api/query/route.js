/**
 * Next.js API Route — Proxy to backend /api/query
 * Eliminates CORS by making the request server-side.
 */

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function POST(request) {
    try {
        const body = await request.json();

        const res = await fetch(`${BACKEND_URL}/api/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        const text = await res.text();
        try {
            const data = JSON.parse(text);
            return Response.json(data, { status: res.status });
        } catch {
            return Response.json(
                { detail: `Backend error (${res.status}): ${text.slice(0, 200)}` },
                { status: res.status }
            );
        }
    } catch (err) {
        return Response.json(
            { detail: `Backend unreachable: ${err.message}` },
            { status: 502 }
        );
    }
}
