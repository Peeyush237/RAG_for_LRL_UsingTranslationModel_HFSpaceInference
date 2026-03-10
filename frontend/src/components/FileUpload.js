'use client';

import { useState, useRef } from 'react';

export default function FileUpload({ lang = 'en', onUpload }) {
    const [files, setFiles] = useState([]);
    const [uploading, setUploading] = useState(false);
    const [result, setResult] = useState(null);
    const [dragActive, setDragActive] = useState(false);
    const inputRef = useRef(null);

    const langLabel = lang === 'en' ? 'English' : 'Odia';

    function handleFiles(newFiles) {
        const arr = Array.from(newFiles).filter(
            (f) => f.name.endsWith('.pdf') || f.name.endsWith('.txt')
        );
        setFiles(arr);
        setResult(null);
    }

    async function handleUpload() {
        if (!files.length) return;
        setUploading(true);
        setResult(null);
        try {
            const res = await onUpload(files, lang);
            setResult({ ok: true, msg: res.message || `${res.num_chunks} chunks indexed` });
            setFiles([]);
        } catch (err) {
            setResult({ ok: false, msg: err.message });
        } finally {
            setUploading(false);
        }
    }

    return (
        <div style={{ marginBottom: 'var(--s5)' }}>
            <p className="label">{langLabel} Documents</p>

            <div
                className={`upload-zone ${dragActive ? 'drag-over' : ''}`}
                onClick={() => inputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
                onDragLeave={() => setDragActive(false)}
                onDrop={(e) => { e.preventDefault(); setDragActive(false); handleFiles(e.dataTransfer.files); }}
            >
                <input
                    ref={inputRef}
                    type="file"
                    multiple
                    accept=".pdf,.txt"
                    style={{ display: 'none' }}
                    onChange={(e) => handleFiles(e.target.files)}
                />
                <p style={{ fontSize: '0.8rem', color: 'var(--warm-gray)' }}>
                    {files.length ? `${files.length} file(s) selected` : 'Drop PDF or TXT here'}
                </p>
            </div>

            {files.length > 0 && (
                <>
                    <div style={{ margin: 'var(--s2) 0' }}>
                        {files.map((f, i) => (
                            <p key={i} style={{ fontSize: '0.75rem', color: 'var(--stone)', padding: '0.2rem 0' }}>
                                {f.name} — {(f.size / 1024).toFixed(0)} KB
                            </p>
                        ))}
                    </div>
                    <button
                        className="btn btn-sm btn-filled"
                        style={{ width: '100%' }}
                        onClick={handleUpload}
                        disabled={uploading}
                    >
                        {uploading ? (
                            <span style={{ display: 'flex', alignItems: 'center', gap: 'var(--s2)' }}>
                                <span className="spinner" /> Indexing
                            </span>
                        ) : (
                            `Build ${langLabel} Index`
                        )}
                    </button>
                </>
            )}

            {result && (
                <p style={{
                    marginTop: 'var(--s2)',
                    fontSize: '0.75rem',
                    color: result.ok ? 'var(--success-green)' : 'var(--error-red)',
                }}>
                    {result.msg}
                </p>
            )}
        </div>
    );
}
