import { holdPosition } from './boardGeometry'

function routeSegments(candidate) {
  if (!candidate) return []
  const { start_holds: starts, middle_holds: middle, end_holds: ends } = candidate
  if (!middle.length) {
    return starts.flatMap((start) => ends.map((end) => [start, end]))
  }
  return [
    ...starts.map((start) => [start, middle[0]]),
    ...middle.slice(0, -1).map((hold, index) => [hold, middle[index + 1]]),
    ...ends.map((end) => [middle[middle.length - 1], end]),
  ]
}

function markerFor(label, candidate) {
  if (!candidate) return null
  if (candidate.start_holds.includes(label)) return 'S'
  if (candidate.end_holds.includes(label)) return 'T'
  const index = candidate.middle_holds.indexOf(label)
  return index >= 0 ? String(index + 1) : null
}

function MoonBoard({
  boardHolds,
  candidate,
  startHolds,
  endHolds,
  selectionMode,
  onHoldSelect,
}) {
  const routeHolds = new Set(candidate?.holds || [])
  const segments = routeSegments(candidate)

  return (
    <section className="panel board-panel" aria-labelledby="board-title">
      <div className="board-heading">
        <div>
          <span className="eyebrow">MoonBoard 2016 · 40°</span>
          <h2 id="board-title">Movement canvas</h2>
        </div>
        <div className="board-legend" aria-label="Route color key">
          <span><i className="legend-dot start" /> Start</span>
          <span><i className="legend-dot move" /> Move</span>
          <span><i className="legend-dot finish" /> Finish</span>
        </div>
      </div>

      <div className="moonboard-frame" aria-describedby="board-instructions">
        <img
          src="/data/moonboard2016Background.jpg"
          alt="MoonBoard 2016 hold layout, columns A to K and rows 1 to 18"
          draggable="false"
        />

        <svg className="route-lines" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          {segments.map(([from, to], index) => {
            const start = holdPosition(from)
            const finish = holdPosition(to)
            return (
              <line
                key={`${from}-${to}-${index}`}
                x1={start.left}
                y1={start.top}
                x2={finish.left}
                y2={finish.top}
              />
            )
          })}
        </svg>

        <div className="hold-layer">
          {boardHolds.map((hold) => {
            const position = holdPosition(hold)
            const isStart = startHolds.includes(hold)
            const isEnd = endHolds.includes(hold)
            const isRoute = routeHolds.has(hold)
            const marker = markerFor(hold, candidate)
            const classes = [
              'board-hold',
              isStart ? 'is-start' : '',
              isEnd ? 'is-finish' : '',
              isRoute && !isStart && !isEnd ? 'is-route' : '',
            ].filter(Boolean).join(' ')
            return (
              <button
                key={hold}
                type="button"
                className={classes}
                style={{ left: `${position.left}%`, top: `${position.top}%` }}
                aria-label={`${hold}${isStart ? ', selected start' : ''}${isEnd ? ', selected finish' : ''}. Set as ${selectionMode}.`}
                aria-pressed={selectionMode === 'start' ? isStart : isEnd}
                tabIndex={isStart || isEnd ? 0 : -1}
                onClick={() => onHoldSelect(hold)}
              >
                <span aria-hidden="true">{marker}</span>
              </button>
            )
          })}
        </div>
      </div>

      <div className="board-footer">
        <span>{boardHolds.length} available holds</span>
        <span>{candidate ? `${candidate.holds.length} holds in selected route` : 'Select endpoints to begin'}</span>
      </div>
    </section>
  )
}

export default MoonBoard
