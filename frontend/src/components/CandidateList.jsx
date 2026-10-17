import { AlertTriangle, Check, Route } from 'lucide-react'

const sourceLabel = (source) => source === 'retrieval_fallback'
  ? 'Conservative match'
  : 'Model alternative'

function CandidateList({ candidates, selectedIndex, onSelect }) {
  return (
    <section className="panel candidates-panel" aria-labelledby="candidates-title">
      <div className="section-heading compact">
        <div>
          <span className="eyebrow">Compare</span>
          <h2 id="candidates-title">Candidate routes</h2>
        </div>
        <span className="step-badge">02</span>
      </div>

      {!candidates.length ? (
        <div className="empty-state">
          <Route size={28} aria-hidden="true" />
          <p>Your generated routes will appear here with movement diagnostics.</p>
        </div>
      ) : (
        <div className="candidate-stack">
          {candidates.map((candidate, index) => {
            const selected = selectedIndex === index
            const warnings = candidate.diagnostics.warnings || []
            return (
              <button
                type="button"
                key={`${candidate.source}-${candidate.rank}`}
                className={selected ? 'candidate-card is-selected' : 'candidate-card'}
                onClick={() => onSelect(index)}
                aria-pressed={selected}
              >
                <div className="candidate-title-row">
                  <span className="candidate-rank">0{candidate.rank}</span>
                  <div>
                    <strong>{sourceLabel(candidate.source)}</strong>
                    <span>{candidate.grade} · {candidate.holds.length} holds</span>
                  </div>
                  {selected && <Check className="selected-check" size={18} aria-label="Selected" />}
                </div>
                <div className="route-sequence" aria-label={`Route sequence: ${candidate.holds.join(', ')}`}>
                  {candidate.holds.join(' · ')}
                </div>
                <div className="candidate-metrics">
                  <span>Longest move <b>{candidate.diagnostics.maximum_move_distance.toFixed(1)}</b></span>
                  <span>{candidate.diagnostics.novel_route ? 'Novel route' : 'Training match'}</span>
                </div>
                {warnings.length > 0 && (
                  <div className="candidate-warning">
                    <AlertTriangle size={15} aria-hidden="true" /> {warnings.join(' · ')}
                  </div>
                )}
              </button>
            )
          })}
        </div>
      )}
      <p className="safety-note">
        Geometry checks improve plausibility, but routes are not certified safe or climbable. Review on the wall.
      </p>
    </section>
  )
}

export default CandidateList
