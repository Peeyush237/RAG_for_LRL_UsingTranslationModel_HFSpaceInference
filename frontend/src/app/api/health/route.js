/**
 * Next.js API Route — Proxy to backend /api/health
 */

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function GET() {
    try {
        const res = await fetch(`${BACKEND_URL}/api/health`);
        const data = await res.json();
        return Response.json(data, { status: res.status });
    } catch (err) {
        return Response.json(
            { detail: `Backend unreachable: ${err.message}` },
            { status: 502 }
        );
    }
}
