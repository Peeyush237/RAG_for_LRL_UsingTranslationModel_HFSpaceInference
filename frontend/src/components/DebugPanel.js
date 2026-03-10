'use client';

import { useState } from 'react';

export default function DebugPanel({ title, debugSteps, chunks, mode = 'lb' }) {
    const [open, setOpen] = useState(false);

    if (!debugSteps?.length && !chunks?.length) return null;

    const accentColor = mode === 'lb' ? 'var(--near-black)' : 'var(--gold)';

    return (
        <div style={{ marginTop: 'var(--s4)' }}>
            <div className="expander-trigger" onClick={() => setOpen(!open)}>
                <h4>{title}</h4>
                <span className={`expander-arrow ${open ? 'open' : ''}`}>▾</span>
            </div>

            <div className={`expander-body ${open ? 'open' : ''}`}>
                <div style={{ padding: 'var(--s4) 0' }}>
                    {debugSteps?.map((step, i) => (
                        <div key={i} style={{ marginBottom: 'var(--s4)' }}>
                            <p style={{
                                fontSize: '0.7rem',
                                fontWeight: 600,
                                letterSpacing: '0.06em',
                                textTransform: 'uppercase',
                                color: 'var(--warm-gray)',
                                marginBottom: 'var(--s1)',
                            }}>
                                {step.step_name}
                            </p>
                            <div style={{
                                fontSize: '0.85rem',
                                color: 'var(--charcoal)',
                                padding: 'var(--s3)',
                                background: 'var(--cream)',
                                fontFamily: 'var(--font-mono)',
                                whiteSpace: 'pre-wrap',
                                wordBreak: 'break-word',
                                lineHeight: 1.65,
                                borderLeft: `2px solid ${accentColor}`,
                            }}>
                                {step.value || '—'}
                            </div>
                        </div>
                    ))}

                    {chunks?.length > 0 && (
                        <div>
                            <p style={{
                                fontSize: '0.7rem',
                                fontWeight: 600,
                                letterSpacing: '0.06em',
                                textTransform: 'uppercase',
                                color: 'var(--warm-gray)',
                                marginBottom: 'var(--s2)',
                            }}>
                                Retrieved Chunks
                            </p>
                            {chunks.map((chunk, i) => (
                                <div key={i} style={{
                                    marginBottom: 'var(--s3)',
                                    padding: 'var(--s3)',
                                    background: 'var(--cream)',
                                    borderLeft: `2px solid ${accentColor}`,
                                }}>
                                    <p style={{
                                        fontSize: '0.65rem',
                                        fontWeight: 600,
                                        color: 'var(--warm-gray)',
                                        letterSpacing: '0.04em',
                                        textTransform: 'uppercase',
                                        marginBottom: 'var(--s1)',
                                    }}>
                                        Chunk {i + 1} · Score {chunk.score?.toFixed(4)}{chunk.source ? ` · ${chunk.source}` : ''}
                                    </p>
                                    <p style={{ fontSize: '0.85rem', color: 'var(--charcoal)', lineHeight: 1.65 }}>
                                        {chunk.text}
                                    </p>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
