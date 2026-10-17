import { AlertTriangle, Check, Route } from 'lucide-react'
import { displayHoldLabel, parseHoldLabel } from './boardGeometry'

const LONG_MOVE_WARNING = 'one or more moves are unusually long for this grade'

const sourceLabel = (source) => source === 'retrieval_fallback'
  ? 'Conservative match'
  : 'Model alternative'

const gridDistance = (left, right) => {
  const leftPoint = parseHoldLabel(left)
  const rightPoint = parseHoldLabel(right)
  return Math.hypot(leftPoint.x - rightPoint.x, leftPoint.y - rightPoint.y)
}

const closestHold = (holds, target) => holds.reduce((closest, hold) => (
  gridDistance(hold, target) < gridDistance(closest, target) ? hold : closest
))

function candidateMoveDetails(candidate) {
  const pairs = []
  const middle = candidate.middle_holds || []
  if (middle.length > 0) {
    pairs.push([closestHold(candidate.start_holds, middle[0]), middle[0]])
    for (let index = 0; index < middle.length - 1; index += 1) {
      pairs.push([middle[index], middle[index + 1]])
    }
    pairs.push([middle.at(-1), closestHold(candidate.end_holds, middle.at(-1))])
  } else {
    const endpointPairs = candidate.start_holds.flatMap((start) => (
      candidate.end_holds.map((end) => [start, end])
    ))
    pairs.push(endpointPairs.reduce((closest, pair) => (
      gridDistance(...pair) < gridDistance(...closest) ? pair : closest
    )))
  }

  return pairs.map(([from, to], index) => ({
    from,
    to,
    distance: candidate.diagnostics.move_distances?.[index] ?? gridDistance(from, to),
  }))
}

function warningItems(candidate, profile) {
  const warnings = candidate.diagnostics.warnings || []
  const items = warnings
    .filter((warning) => warning !== LONG_MOVE_WARNING)
    .map((warning) => ({ title: warning }))

  if (!warnings.includes(LONG_MOVE_WARNING) || !profile?.soft_move_limit) {
    if (warnings.includes(LONG_MOVE_WARNING)) items.unshift({ title: LONG_MOVE_WARNING })
    return items
  }

  const largeMoves = candidateMoveDetails(candidate)
    .filter((move) => move.distance > profile.soft_move_limit)
  largeMoves.reverse().forEach((move) => {
    items.unshift({
      title: `Large move · ${displayHoldLabel(move.from)} → ${displayHoldLabel(move.to)} · ${move.distance.toFixed(1)} grid units`,
      detail: `Longer than 95% of recorded ${candidate.grade} moves`,
    })
  })
  return items
}

function CandidateList({ candidates, selectedIndex, onSelect, profiles = {} }) {
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
            const warnings = warningItems(candidate, profiles[candidate.grade])
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
                <div className="route-hold-groups" aria-label={`Included holds: starts ${candidate.start_holds.map(displayHoldLabel).join(', ')}; other holds ${candidate.middle_holds.map(displayHoldLabel).join(', ')}; finishes ${candidate.end_holds.map(displayHoldLabel).join(', ')}`}>
                  <span><i className="hold-key start" />Start {candidate.start_holds.map(displayHoldLabel).join(', ')}</span>
                  {candidate.middle_holds.length > 0 && (
                    <span><i className="hold-key move" />Holds {candidate.middle_holds.map(displayHoldLabel).join(', ')}</span>
                  )}
                  <span><i className="hold-key finish" />Finish {candidate.end_holds.map(displayHoldLabel).join(', ')}</span>
                </div>
                <div className="candidate-metrics">
                  <span>Longest grid gap <b>{candidate.diagnostics.maximum_move_distance.toFixed(1)}</b></span>
                  <span>{candidate.diagnostics.novel_route ? 'Novel route' : 'Training match'}</span>
                </div>
                {warnings.length > 0 && (
                  <div className="candidate-warnings">
                    {warnings.map((warning) => (
                      <div className="candidate-warning" key={`${warning.title}-${warning.detail || ''}`}>
                        <AlertTriangle size={15} aria-hidden="true" />
                        <span>
                          <strong>{warning.title}</strong>
                          {warning.detail && <small>{warning.detail}</small>}
                        </span>
                      </div>
                    ))}
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
