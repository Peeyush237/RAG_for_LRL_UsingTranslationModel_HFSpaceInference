import './globals.css';

export const metadata = {
    title: 'LinguaBridge — RAG for Low-Resource Languages',
    description: 'Cross-lingual retrieval-augmented QA for Odia and English, powered by fine-tuned IndicTrans2 and advanced hybrid retrieval.',
};

export default function RootLayout({ children }) {
    return (
        <html lang="en">
            <body>{children}</body>
        </html>
    );
}
