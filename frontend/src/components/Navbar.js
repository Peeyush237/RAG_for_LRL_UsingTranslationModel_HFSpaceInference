'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Navbar() {
    const pathname = usePathname();

    return (
        <nav style={{
            padding: '1.2rem 0',
            borderBottom: '1px solid rgba(0,0,0,0.06)',
        }}>
            <div className="container" style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
            }}>
                <Link href="/" style={{
                    fontFamily: 'var(--font-display)',
                    fontSize: '1.4rem',
                    fontWeight: 500,
                    color: 'var(--near-black)',
                    letterSpacing: '-0.01em',
                    borderBottom: 'none',
                }}>
                    LinguaBridge
                </Link>

                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--s6)' }}>
                    <Link
                        href="/"
                        style={{
                            fontSize: '0.8rem',
                            fontWeight: 500,
                            letterSpacing: '0.04em',
                            textTransform: 'uppercase',
                            color: pathname === '/' ? 'var(--near-black)' : 'var(--warm-gray)',
                            borderBottom: pathname === '/' ? '1px solid var(--near-black)' : 'none',
                            paddingBottom: '2px',
                        }}
                    >
                        About
                    </Link>
                    <Link
                        href="/chat"
                        style={{
                            fontSize: '0.8rem',
                            fontWeight: 500,
                            letterSpacing: '0.04em',
                            textTransform: 'uppercase',
                            color: pathname === '/chat' ? 'var(--near-black)' : 'var(--warm-gray)',
                            borderBottom: pathname === '/chat' ? '1px solid var(--near-black)' : 'none',
                            paddingBottom: '2px',
                        }}
                    >
                        Query
                    </Link>
                    <a
                        href="https://github.com/Peeyush237/RAG_for_LowResourceLanguage_using_FineTuned_translationModel"
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{
                            fontSize: '0.8rem',
                            fontWeight: 500,
                            letterSpacing: '0.04em',
                            textTransform: 'uppercase',
                            color: 'var(--warm-gray)',
                            borderBottom: 'none',
                        }}
                    >
                        GitHub ↗
                    </a>
                </div>
            </div>
        </nav>
    );
}
