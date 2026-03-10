'use client';

export default function MetricsCard({ label, value, delta }) {
    let deltaClass = '';
    let deltaPrefix = '';
    if (delta !== undefined && delta !== null) {
        deltaClass = delta > 0 ? 'positive' : delta < 0 ? 'negative' : '';
        deltaPrefix = delta > 0 ? '+' : '';
    }

    return (
        <div className="metric">
            <div className="metric-value">{value}</div>
            <div className="metric-label">{label}</div>
            {delta !== undefined && delta !== null && (
                <div className={`metric-delta ${deltaClass}`}>
                    {deltaPrefix}{typeof delta === 'number' ? delta.toFixed(3) : delta}
                </div>
            )}
        </div>
    );
}
