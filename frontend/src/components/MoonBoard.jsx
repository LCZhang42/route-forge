import { Pencil, RotateCcw } from 'lucide-react'
import { displayHoldLabel, holdPosition } from './boardGeometry'

function MoonBoard({
  boardHolds,
  candidate,
  startHolds,
  endHolds,
  selectableHolds,
  selectionMode,
  onHoldSelect,
  routeEditMode = false,
  routeModified = false,
  movingHold = null,
  onRouteEditToggle,
  onRouteHoldSelect,
  onRouteRestore,
}) {
  const routeHolds = new Set(candidate?.holds || [])
  const routeStarts = new Set(candidate?.start_holds || [])
  const routeFinishes = new Set(candidate?.end_holds || [])
  const selectable = new Set(selectableHolds)

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

      {candidate && (
        <div className={routeEditMode ? 'route-edit-bar is-editing' : 'route-edit-bar'}>
          <div className="route-edit-copy" role="status">
            <strong>{routeModified ? 'Custom route' : 'Generated route'}</strong>
            <span>
              {routeEditMode
                ? movingHold
                  ? `Moving ${displayHoldLabel(movingHold)} — choose its new hold.`
                  : 'Select a colored ring, then choose its new hold.'
                : routeModified
                  ? 'Your changes are shown on the board.'
                  : 'Edit the rings without changing the generated result.'}
            </span>
          </div>
          <div className="route-edit-actions">
            {routeModified && (
              <button type="button" className="route-edit-button secondary" onClick={onRouteRestore}>
                <RotateCcw size={15} aria-hidden="true" /> Restore generated route
              </button>
            )}
            <button
              type="button"
              className={routeEditMode ? 'route-edit-button is-active' : 'route-edit-button'}
              aria-pressed={routeEditMode}
              onClick={onRouteEditToggle}
            >
              <Pencil size={15} aria-hidden="true" /> {routeEditMode ? 'Done editing' : 'Edit route'}
            </button>
          </div>
        </div>
      )}

      <div className={routeEditMode ? 'moonboard-frame is-editing' : 'moonboard-frame'} aria-describedby="board-instructions">
        <img
          src="/data/moonboard2016Background.jpg"
          alt="MoonBoard 2016 hold layout, columns A to K and rows 1 to 18"
          draggable="false"
        />

        <div className="hold-layer">
          {boardHolds.map((hold) => {
            const position = holdPosition(hold)
            const isStart = candidate ? routeStarts.has(hold) : startHolds.includes(hold)
            const isEnd = candidate ? routeFinishes.has(hold) : endHolds.includes(hold)
            const isRoute = routeHolds.has(hold)
            const canSelect = selectable.has(hold)
            const isMoveSource = movingHold === hold
            const displayLabel = displayHoldLabel(hold)
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
              isMoveSource ? 'is-move-source' : '',
            ].filter(Boolean).join(' ')
            const editInstruction = isRoute
              ? 'Select this ring to move it.'
              : movingHold
                ? `Move ${displayHoldLabel(movingHold)} here.`
                : 'Select a colored ring before choosing a destination.'
            return (
              <button
                key={hold}
                type="button"
                className={classes}
                style={{ left: `${position.left}%`, top: `${position.top}%` }}
                aria-label={`${displayLabel}${routeRole ? `, ${routeRole}` : ''}${isStart ? ', selected start' : ''}${isEnd ? ', selected finish' : ''}. ${routeEditMode ? editInstruction : canSelect ? `Set as ${selectionMode}.` : `Not available as ${selectionMode}.`}`}
                aria-pressed={routeEditMode ? isMoveSource : selectionMode === 'start' ? isStart : isEnd}
                disabled={!routeEditMode && !canSelect}
                tabIndex={routeEditMode || isStart || isEnd ? 0 : -1}
                onClick={() => routeEditMode ? onRouteHoldSelect(hold) : onHoldSelect(hold)}
              />
            )
          })}
        </div>
      </div>

      <div className="board-footer">
        <span>{routeEditMode ? 'Route editing is on' : `${selectableHolds.length} available ${selectionMode} positions`}</span>
        <span>{candidate ? `${candidate.holds.length} holds in ${routeModified ? 'custom' : 'selected'} route` : 'Select endpoints to begin'}</span>
      </div>
    </section>
  )
}

export default MoonBoard
