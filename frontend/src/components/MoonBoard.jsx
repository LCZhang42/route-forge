import { holdPosition } from './boardGeometry'

function MoonBoard({
  boardHolds,
  candidate,
  startHolds,
  endHolds,
  selectionMode,
  onHoldSelect,
}) {
  const routeHolds = new Set(candidate?.holds || [])
  const routeStarts = new Set(candidate?.start_holds || [])
  const routeFinishes = new Set(candidate?.end_holds || [])

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

        <div className="hold-layer">
          {boardHolds.map((hold) => {
            const position = holdPosition(hold)
            const isStart = startHolds.includes(hold)
            const isEnd = endHolds.includes(hold)
            const isRoute = routeHolds.has(hold)
            const routeRole = routeStarts.has(hold)
              ? 'route start'
              : routeFinishes.has(hold)
                ? 'route finish'
                : isRoute
                  ? 'route hold'
                  : null
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
                aria-label={`${hold}${routeRole ? `, ${routeRole}` : ''}${isStart ? ', selected start' : ''}${isEnd ? ', selected finish' : ''}. Set as ${selectionMode}.`}
                aria-pressed={selectionMode === 'start' ? isStart : isEnd}
                tabIndex={isStart || isEnd ? 0 : -1}
                onClick={() => onHoldSelect(hold)}
              />
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
