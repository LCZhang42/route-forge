import { useState } from 'react'
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
  onRouteHoldMove,
  onRouteRestore,
}) {
  const [dragSource, setDragSource] = useState(null)
  const [dropTarget, setDropTarget] = useState(null)
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
                ? dragSource
                  ? `Dragging ${displayHoldLabel(dragSource)} — release it on a new hold.`
                  : movingHold
                  ? `Moving ${displayHoldLabel(movingHold)} — choose its new hold.`
                  : 'Drag a colored ring to a new hold.'
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
            const isMoveSource = movingHold === hold || dragSource === hold
            const isDropTarget = dropTarget === hold
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
              isDropTarget ? 'is-drop-target' : '',
            ].filter(Boolean).join(' ')
            const editInstruction = isRoute
              ? 'Drag this ring to another hold. Keyboard: press Enter to pick it up.'
              : movingHold
                ? `Press Enter to move ${displayHoldLabel(movingHold)} here.`
                : 'Drop a colored ring here. Keyboard: select a colored ring first.'
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
                draggable={routeEditMode && isRoute}
                onClick={(event) => {
                  if (!routeEditMode) onHoldSelect(hold)
                  else if (event.detail === 0) onRouteHoldSelect(hold)
                }}
                onDragStart={(event) => {
                  if (!routeEditMode || !isRoute) {
                    event.preventDefault()
                    return
                  }
                  event.dataTransfer.effectAllowed = 'move'
                  event.dataTransfer.setData('text/plain', hold)
                  setDragSource(hold)
                  setDropTarget(null)
                }}
                onDragEnter={(event) => {
                  if (!routeEditMode || isRoute || !dragSource) return
                  event.preventDefault()
                  setDropTarget(hold)
                }}
                onDragOver={(event) => {
                  if (!routeEditMode || isRoute || !dragSource) return
                  event.preventDefault()
                  event.dataTransfer.dropEffect = 'move'
                  if (dropTarget !== hold) setDropTarget(hold)
                }}
                onDragLeave={() => {
                  if (dropTarget === hold) setDropTarget(null)
                }}
                onDrop={(event) => {
                  event.preventDefault()
                  const source = dragSource || event.dataTransfer.getData('text/plain')
                  if (source && source !== hold && !isRoute) onRouteHoldMove(source, hold)
                  setDragSource(null)
                  setDropTarget(null)
                }}
                onDragEnd={() => {
                  setDragSource(null)
                  setDropTarget(null)
                }}
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
