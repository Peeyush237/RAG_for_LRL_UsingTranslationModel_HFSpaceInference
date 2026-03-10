import Link from 'next/link';
import Navbar from '@/components/Navbar';

const PIPELINE_STEPS = [
    'Odia Query',
    'Entity Guard',
    'Translate → EN',
    'Hybrid Retrieve',
    'Rerank',
    'LLM Answer',
    'Translate → OR',
];

export default function HomePage() {
    return (
        <>
            <Navbar />

            {/* ── Hero ─────────────────────────────── */}
            <section style={{
                padding: 'var(--s9) 0 var(--s8)',
                textAlign: 'center',
            }}>
                <div className="container-narrow">
                    <p style={{
                        fontSize: '0.7rem',
                        fontWeight: 600,
                        letterSpacing: '0.12em',
                        textTransform: 'uppercase',
                        color: 'var(--gold)',
                        marginBottom: 'var(--s5)',
                    }}>
                        Cross-Lingual Retrieval-Augmented Generation
                    </p>

                    <h1 style={{ marginBottom: 'var(--s5)' }}>
                        Ask in Odia.<br />
                        <em style={{ fontWeight: 400 }}>Retrieve from English.</em>
                    </h1>

                    <p style={{
                        maxWidth: '520px',
                        margin: '0 auto var(--s6)',
                        fontSize: '1.05rem',
                        lineHeight: 1.75,
                    }}>
                        A fine-tuned IndicTrans2 translation model paired with
                        hybrid semantic search — bridging the gap for low-resource languages.
                    </p>

                    <Link href="/chat" className="btn btn-filled">
                        Open Query Interface
                    </Link>
                </div>
            </section>

            <hr className="rule" style={{ maxWidth: '80px', margin: '0 auto var(--s8)' }} />

            {/* ── Pipeline ─────────────────────────── */}
            <section style={{ padding: '0 0 var(--s8)' }}>
                <div className="container-narrow" style={{ textAlign: 'center' }}>
                    <h2 style={{ marginBottom: 'var(--s7)' }}>The Pipeline</h2>

                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: 'var(--s3)',
                        flexWrap: 'wrap',
                    }}>
                        {PIPELINE_STEPS.map((step, i) => (
                            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 'var(--s3)' }}>
                                <span style={{
                                    fontFamily: 'var(--font-display)',
                                    fontSize: '0.95rem',
                                    fontWeight: 500,
                                    color: 'var(--near-black)',
                                    whiteSpace: 'nowrap',
                                }}>
                                    {step}
                                </span>
                                {i < PIPELINE_STEPS.length - 1 && (
                                    <span style={{ color: 'var(--sand)', fontSize: '0.8rem' }}>→</span>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            <hr className="rule" style={{ maxWidth: '80px', margin: '0 auto var(--s8)' }} />

            {/* ── Two approaches ───────────────────── */}
            <section style={{ padding: '0 0 var(--s8)' }}>
                <div className="container">
                    <div className="columns-2">
                        <div>
                            <p className="tag tag-gold" style={{ marginBottom: 'var(--s4)' }}>LinguaBridge</p>
                            <h3 style={{ marginBottom: 'var(--s4)' }}>Cross-lingual retrieval</h3>
                            <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 'var(--s2)' }}>
                                {[
                                    'Queries in any Indic language',
                                    'Retrieves from English knowledge base',
                                    'Fine-tuned IndicTrans2 translation (BLEU 47)',
                                    'Entity-protected translation pipeline',
                                    'BM25 + FAISS hybrid search with reranking',
                                ].map((item, i) => (
                                    <li key={i} style={{ fontSize: '0.9rem', color: 'var(--stone)' }}>
                                        — {item}
                                    </li>
                                ))}
                            </ul>
                        </div>

                        <div>
                            <p className="tag" style={{ marginBottom: 'var(--s4)' }}>Traditional RAG</p>
                            <h3 style={{ marginBottom: 'var(--s4)' }}>Same-language retrieval</h3>
                            <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 'var(--s2)' }}>
                                {[
                                    'Odia queries on Odia documents',
                                    'No translation overhead',
                                    'Simpler, fewer pipeline steps',
                                    'Limited by Odia content availability',
                                    'Weaker cross-lingual understanding',
                                ].map((item, i) => (
                                    <li key={i} style={{ fontSize: '0.9rem', color: 'var(--stone)' }}>
                                        — {item}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>
                </div>
            </section>

            <hr className="rule" style={{ maxWidth: '80px', margin: '0 auto var(--s8)' }} />

            {/* ── CTA ──────────────────────────────── */}
            <section style={{ padding: '0 0 var(--s9)', textAlign: 'center' }}>
                <div className="container-narrow">
                    <h2 style={{ marginBottom: 'var(--s4)' }}>Try it yourself</h2>
                    <p style={{ marginBottom: 'var(--s6)', maxWidth: '420px', margin: '0 auto var(--s6)' }}>
                        Upload your documents and ask questions in Odia.
                        Compare both approaches side by side.
                    </p>
                    <Link href="/chat" className="btn btn-gold">
                        Open Query Interface
                    </Link>
                </div>
            </section>

            {/* ── Footer ───────────────────────────── */}
            <footer style={{
                padding: 'var(--s5) 0',
                borderTop: '1px solid rgba(0,0,0,0.06)',
                textAlign: 'center',
            }}>
                <p style={{ fontSize: '0.75rem', color: 'var(--warm-gray)' }}>
                    IIIT Nagpur · Semester VI CSA ·
                    Peeyush Mishra, Divyal Surse, Sandesh Charhate ·
                    Guide: Mr. Amol Bhopale
                </p>
            </footer>
        </>
    );
}
