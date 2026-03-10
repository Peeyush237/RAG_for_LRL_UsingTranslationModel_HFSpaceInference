/**
 * api.js — Frontend API client for the LinguaBridge backend.
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Check backend health status.
 */
export async function getHealth() {
    const res = await fetch(`${API_URL}/api/health`);
    if (!res.ok) throw new Error('Backend unreachable');
    return res.json();
}

/**
 * Run RAG query pipeline.
 */
export async function queryPipeline({ question, pipelineMode = 'both', topK = 5 }) {
    const res = await fetch(`${API_URL}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            question,
            pipeline_mode: pipelineMode,
            top_k: topK,
        }),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
}

/**
 * Upload and ingest documents.
 */
export async function ingestDocuments(files, lang = 'en') {
    const formData = new FormData();
    for (const file of files) {
        formData.append('files', file);
    }
    formData.append('lang', lang);

    const res = await fetch(`${API_URL}/api/ingest`, {
        method: 'POST',
        body: formData,
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
        throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
}

/**
 * Translate text.
 */
export async function translateText(text, sourceLang, targetLang) {
    const res = await fetch(`${API_URL}/api/translate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            text,
            source_lang: sourceLang,
            target_lang: targetLang,
        }),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Translation failed' }));
        throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
}
