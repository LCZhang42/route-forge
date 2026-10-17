import { Flag, MapPin, Sparkles } from 'lucide-react'
import GradeSelector from './GradeSelector'
import { displayHoldLabel } from './boardGeometry'

const ROUTE_PRESETS = [
  {
    name: 'Balanced',
    detail: 'Two-hand start · steady center finish',
    grade: '6C+',
    startHolds: ['A4', 'F4'],
    endHolds: ['G17'],
  },
  {
    name: 'Center pull',
    detail: 'Single start · direct finish',
    grade: '7A',
    startHolds: ['D5'],
    endHolds: ['G17'],
  },
  {
    name: 'Wide launch',
    detail: 'Wide start · longer movement',
    grade: '7B',
    startHolds: ['B5', 'K5'],
    endHolds: ['H17'],
  },
]

function HoldSelect({ id, label, values, boardHolds, excluded, onChange }) {
  const update = (index, value) => {
    if (!value && index === 1) {
      onChange(values.slice(0, 1))
      return
    }
    const next = [...values]
    next[index] = value
    onChange(next.filter(Boolean))
  }

  const optionsFor = (index) => boardHolds.filter(
    (hold) => hold === values[index]
      || (!values.includes(hold) && !excluded.includes(hold)),
  )

  return (
    <div className="hold-select-group">
      <div className="field-label-row">
        <label htmlFor={`${id}-primary`}>{label}</label>
        <span>1–2 holds</span>
      </div>
      <div className="hold-selects">
        <select
          id={`${id}-primary`}
          value={values[0] || ''}
          onChange={(event) => update(0, event.target.value)}
          required
        >
          <option value="" disabled>Select a hold</option>
          {optionsFor(0).map((hold) => (
            <option key={hold} value={hold}>{displayHoldLabel(hold)}</option>
          ))}
        </select>
        <select
          aria-label={`Optional second ${label.toLowerCase()}`}
          value={values[1] || ''}
          onChange={(event) => update(1, event.target.value)}
        >
          <option value="">No second hold</option>
          {optionsFor(1).map((hold) => (
            <option key={hold} value={hold}>{displayHoldLabel(hold)}</option>
          ))}
        </select>
      </div>
    </div>
  )
}

function Controls({
  capabilities,
  selectedGrade,
  onGradeChange,
  startHolds,
  endHolds,
  onStartChange,
  onEndChange,
  selectionMode,
  onSelectionModeChange,
  seed,
  onSeedChange,
  temperature,
  onTemperatureChange,
  onApplyPreset,
  onGenerate,
  isGenerating,
}) {
  const ready = Boolean(selectedGrade && startHolds.length && endHolds.length)

  return (
    <section className="panel setup-panel" aria-labelledby="setup-title">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Route brief</span>
          <h2 id="setup-title">Set the problem</h2>
        </div>
        <span className="step-badge">01</span>
      </div>

      <div className="preset-row" aria-label="Route presets">
        {ROUTE_PRESETS.map((preset) => (
          <button
            className="preset-button"
            type="button"
            key={preset.name}
            onClick={() => onApplyPreset(preset)}
            title={preset.detail}
          >
            {preset.name}
          </button>
        ))}
      </div>

      <GradeSelector
        grades={capabilities.supported_grades}
        selectedGrade={selectedGrade}
        onGradeChange={onGradeChange}
        disabled={isGenerating}
      />

      <div className="board-mode" aria-label="Board selection mode">
        <button
          type="button"
          className={selectionMode === 'start' ? 'mode-button start is-active' : 'mode-button start'}
          aria-pressed={selectionMode === 'start'}
          onClick={() => onSelectionModeChange('start')}
        >
          <MapPin size={17} aria-hidden="true" /> Select start
        </button>
        <button
          type="button"
          className={selectionMode === 'end' ? 'mode-button finish is-active' : 'mode-button finish'}
          aria-pressed={selectionMode === 'end'}
          onClick={() => onSelectionModeChange('end')}
        >
          <Flag size={17} aria-hidden="true" /> Select finish
        </button>
      </div>
      <p className="mode-help" id="board-instructions">
        Choose a mode, then select an available ring on the board. Starts use observed 2016 start positions; finishes use top-row holds.
      </p>

      <div className="endpoint-fields">
        <HoldSelect
          id="start"
          label="Start holds"
          values={startHolds}
          boardHolds={capabilities.endpoint_holds.start}
          excluded={endHolds}
          onChange={onStartChange}
        />
        <HoldSelect
          id="finish"
          label="Finish holds"
          values={endHolds}
          boardHolds={capabilities.endpoint_holds.end}
          excluded={startHolds}
          onChange={onEndChange}
        />
      </div>

      <details className="advanced-settings">
        <summary>Generation settings</summary>
        <div className="advanced-grid">
          <label>
            Reproducible seed
            <input
              type="number"
              min="0"
              max="2147483647"
              value={seed ?? ''}
              placeholder="Random"
              onChange={(event) => onSeedChange(
                event.target.value === '' ? null : Number(event.target.value),
              )}
            />
          </label>
          <label>
            Variety <output>{temperature.toFixed(1)}</output>
            <input
              type="range"
              min="0.2"
              max="1.3"
              step="0.1"
              value={temperature}
              onChange={(event) => onTemperatureChange(Number(event.target.value))}
            />
          </label>
        </div>
      </details>

      <button
        className="generate-button"
        type="button"
        disabled={!ready || isGenerating}
        onClick={onGenerate}
      >
        <Sparkles size={19} aria-hidden="true" />
        {isGenerating ? 'Building three routes…' : 'Forge routes'}
      </button>
    </section>
  )
}

export default Controls
