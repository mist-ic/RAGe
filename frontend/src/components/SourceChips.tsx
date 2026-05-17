import { useState } from 'react'
import type { Source } from '../types'

interface SourceChipsProps {
  sources: Source[]
}

const LABEL_COLORS: Record<string, string> = {
  correct:   '#22c55e',  // green
  ambiguous: '#f59e0b',  // amber
  incorrect: '#ef4444',  // red (shouldn't normally appear in kept chunks)
}

export function SourceChips({ sources }: SourceChipsProps) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null)

  if (!sources || sources.length === 0) return null

  const scoreClass = (score: number) => {
    if (score >= 0.7) return 'high'
    if (score >= 0.5) return 'mid'
    return 'low'
  }

  return (
    <div className="sources">
      {sources.map((src, i) => (
        <div
          key={i}
          className="source-chip"
          onMouseEnter={() => setHoveredIdx(i)}
          onMouseLeave={() => setHoveredIdx(null)}
        >
          <span className={`source-score ${scoreClass(src.relevance_score)}`} />
          <span>{src.document}</span>
          {src.page && <span style={{ opacity: 0.6 }}>p.{src.page}</span>}
          {src.crag_label && (
            <span style={{
              fontSize: 10,
              fontWeight: 600,
              color: LABEL_COLORS[src.crag_label] ?? 'var(--text-muted)',
              textTransform: 'uppercase',
              letterSpacing: '0.04em',
            }}>
              {src.crag_label}
            </span>
          )}
          {hoveredIdx === i && src.text_preview && (
            <div className="source-tooltip">
              {src.section && (
                <div style={{ color: '#6c63ff', fontWeight: 600, marginBottom: 4 }}>
                  {src.section}
                </div>
              )}
              <div style={{ opacity: 0.8 }}>{src.text_preview}</div>
              <div style={{ marginTop: 6, opacity: 0.5 }}>
                Score: {src.relevance_score.toFixed(3)}
                {src.crag_label && ` · CRAG: ${src.crag_label}`}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
