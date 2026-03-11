'use client';

import { useState } from 'react';
import Navbar from '@/components/Navbar';
import FileUpload from '@/components/FileUpload';
import DebugPanel from '@/components/DebugPanel';
import MetricsCard from '@/components/MetricsCard';
import { queryPipeline, ingestDocuments } from '@/lib/api';

export default function ChatPage() {
    const [question, setQuestion] = useState('');
    const [topK, setTopK] = useState(5);
    const [pipelineMode, setPipelineMode] = useState('both');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');

    async function handleQuery() {
        if (!question.trim()) return;
        setLoading(true);
        setError('');
        setResult(null);
        try {
            const data = await queryPipeline({
                question: question.trim(),
                pipelineMode,
                topK,
            });
            setResult(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }

    const lb = result?.linguabridge;
    const trad = result?.traditional;

    return (
        <>
            <Navbar />

            <div className="app-layout" style={{
                display: 'grid',
                gridTemplateColumns: '240px 1fr',
                gap: 'var(--s7)',
                maxWidth: '1200px',
                margin: '0 auto',
                padding: 'var(--s6) var(--s5) var(--s8)',
                minHeight: 'calc(100vh - 60px)',
            }}>
                {/* ── Sidebar ──────────────────────── */}
                <aside className="sidebar" style={{
                    position: 'sticky',
                    top: 'var(--s6)',
                    height: 'calc(100vh - 5rem)',
                    overflowY: 'auto',
                    paddingRight: 'var(--s4)',
                    borderRight: '1px solid rgba(0,0,0,0.06)',
                }}>
                    <p className="label" style={{ marginBottom: 'var(--s4)' }}>Settings</p>

                    {/* Pipeline Mode */}
                    <div style={{ marginBottom: 'var(--s5)' }}>
                        <p className="label">Pipeline</p>
                        <select
                            className="input"
                            value={pipelineMode}
                            onChange={(e) => setPipelineMode(e.target.value)}
                            style={{ fontSize: '0.85rem' }}
                        >
                            <option value="both">Both — Compare</option>
                            <option value="linguabridge">LinguaBridge Only</option>
                            <option value="traditional">Traditional RAG Only</option>
                        </select>
                    </div>

                    {/* Top K */}
                    <div style={{ marginBottom: 'var(--s5)' }}>
                        <p className="label">Top-K Chunks: {topK}</p>
                        <input
                            type="range"
                            min="1"
                            max="10"
                            value={topK}
                            onChange={(e) => setTopK(Number(e.target.value))}
                            style={{ width: '100%', accentColor: 'var(--gold)' }}
                        />
                    </div>

                    <hr className="rule-light" />

                    {/* Document Upload */}
                    <FileUpload lang="en" onUpload={ingestDocuments} />
                    <FileUpload lang="od" onUpload={ingestDocuments} />
                </aside>

                {/* ── Main ─────────────────────────── */}
                <main>
                    {/* Query box */}
                    <div style={{ marginBottom: 'var(--s7)' }}>
                        <h2 style={{ marginBottom: 'var(--s4)' }}>
                            Query
                        </h2>

                        <textarea
                            className="textarea"
                            placeholder="ଏଠାରେ ଆପଣଙ୍କ ପ୍ରଶ୍ନ ଲେଖନ୍ତୁ..."
                            value={question}
                            onChange={(e) => setQuestion(e.target.value)}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter' && !e.shiftKey) {
                                    e.preventDefault();
                                    handleQuery();
                                }
                            }}
                            style={{ minHeight: '100px', marginBottom: 'var(--s3)' }}
                        />

                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--s4)' }}>
                            <button
                                className="btn btn-filled"
                                onClick={handleQuery}
                                disabled={loading || !question.trim()}
                            >
                                {loading ? (
                                    <span style={{ display: 'flex', alignItems: 'center', gap: 'var(--s2)' }}>
                                        <span className="spinner" /> Searching
                                    </span>
                                ) : (
                                    'Search & Answer'
                                )}
                            </button>

                            {error && (
                                <span style={{ color: 'var(--error-red)', fontSize: '0.85rem' }}>
                                    {error}
                                </span>
                            )}
                        </div>

                        {loading && (
                            <div style={{
                                marginTop: 'var(--s3)',
                                fontSize: '0.85rem',
                                color: 'var(--stone)',
                                fontStyle: 'italic',
                                animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                            }}>
                                ⏳ Inference in progress. Cross-lingual retrieval and translation may take 30-60 seconds on free-tier APIs. Please be patient...
                            </div>
                        )}
                    </div>

                    {/* ── Results ────────────────────── */}
                    {result && (
                        <>
                            {/* Answers */}
                            <div className="columns-2" style={{ marginBottom: 'var(--s7)' }}>
                                {lb && (
                                    <div>
                                        <p className="label" style={{ marginBottom: 'var(--s2)' }}>LinguaBridge</p>
                                        <div className="answer-block">
                                            {lb.answer}
                                        </div>
                                        {lb.answer_en && (
                                            <div style={{
                                                marginTop: 'var(--s3)',
                                                padding: 'var(--s3) var(--s4)',
                                                background: 'var(--cream)',
                                                borderLeft: '2px solid var(--sand)',
                                                fontSize: '0.85rem',
                                                color: 'var(--stone)',
                                            }}>
                                                <small style={{ display: 'block', marginBottom: 'var(--s1)' }}>English intermediate</small>
                                                {lb.answer_en}
                                            </div>
                                        )}
                                    </div>
                                )}

                                {trad && (
                                    <div>
                                        <p className="label" style={{ marginBottom: 'var(--s2)' }}>Traditional RAG</p>
                                        <div className="answer-block-alt">
                                            {trad.answer}
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Metrics */}
                            <div style={{ marginBottom: 'var(--s6)' }}>
                                <p className="label" style={{ marginBottom: 'var(--s3)' }}>Performance</p>
                                <div className="columns-4" style={{
                                    borderTop: '1px solid rgba(0,0,0,0.06)',
                                    borderBottom: '1px solid rgba(0,0,0,0.06)',
                                    padding: 'var(--s3) 0',
                                }}>
                                    {lb && (
                                        <>
                                            <MetricsCard
                                                label="LB Time"
                                                value={`${lb.response_time}s`}
                                            />
                                            <MetricsCard
                                                label="LB Retrieval"
                                                value={lb.retrieval_score?.toFixed(3) || '—'}
                                                delta={trad ? lb.retrieval_score - trad.retrieval_score : null}
                                            />
                                        </>
                                    )}
                                    {trad && (
                                        <>
                                            <MetricsCard
                                                label="Trad. Time"
                                                value={`${trad.response_time}s`}
                                            />
                                            <MetricsCard
                                                label="Trad. Retrieval"
                                                value={trad.retrieval_score?.toFixed(3) || '—'}
                                            />
                                        </>
                                    )}
                                </div>
                            </div>

                            {/* Evaluation */}
                            {result.evaluation && (
                                <div style={{ marginBottom: 'var(--s6)' }}>
                                    <p className="label" style={{ marginBottom: 'var(--s3)' }}>LLM Evaluation</p>
                                    <div style={{
                                        padding: 'var(--s5)',
                                        background: 'var(--cream)',
                                        borderLeft: '2px solid var(--gold)',
                                        whiteSpace: 'pre-wrap',
                                        lineHeight: 1.75,
                                        fontSize: '0.9rem',
                                        color: 'var(--charcoal)',
                                    }}>
                                        {result.evaluation}
                                    </div>
                                </div>
                            )}

                            {/* Debug */}
                            {lb && (
                                <DebugPanel
                                    title="LinguaBridge Pipeline Trace"
                                    debugSteps={lb.debug_steps}
                                    chunks={lb.retrieved_chunks}
                                    mode="lb"
                                />
                            )}
                            {trad && (
                                <DebugPanel
                                    title="Traditional RAG Pipeline Trace"
                                    debugSteps={trad.debug_steps}
                                    chunks={trad.retrieved_chunks}
                                    mode="trad"
                                />
                            )}
                        </>
                    )}

                    {/* Empty state */}
                    {!result && !loading && (
                        <div style={{
                            padding: 'var(--s9) var(--s6)',
                            textAlign: 'center',
                        }}>
                            <h3 style={{
                                fontFamily: 'var(--font-display)',
                                fontWeight: 400,
                                fontSize: '1.6rem',
                                color: 'var(--warm-gray)',
                                marginBottom: 'var(--s3)',
                                fontStyle: 'italic',
                            }}>
                                Upload documents, then ask.
                            </h3>
                            <p style={{ fontSize: '0.85rem', color: 'var(--sand)', maxWidth: '380px', margin: '0 auto' }}>
                                Add English or Odia files using the sidebar,
                                then type your question in Odia above.
                            </p>
                        </div>
                    )}
                </main>
            </div>
        </>
    );
}
