import { useEffect, useRef, useState } from 'react'
import { Activity, Check, Copy, Mountain, RefreshCw } from 'lucide-react'
import MoonBoard from './components/MoonBoard'
import Controls from './components/Controls'
import CandidateList from './components/CandidateList'
import {
  displayHoldLabel,
  normalizeEndpointHolds,
  physicalBoardHolds,
} from './components/boardGeometry'
import {
  generateRoutes,
  getApiErrorMessage,
  getModelCapabilities,
} from './api/client'
import { readRouteState, writeRouteState } from './state'

const initialRouteState = readRouteState(
  typeof window === 'undefined' ? '' : window.location.search,
)

function App() {
  const [capabilities, setCapabilities] = useState(null)
  const [capabilityAttempt, setCapabilityAttempt] = useState(0)
  const [serviceState, setServiceState] = useState('loading')
  const [serviceMessage, setServiceMessage] = useState('Connecting to the route service…')
  const initialized = useRef(false)

  const [selectedGrade, setSelectedGrade] = useState(initialRouteState.grade)
  const [startHolds, setStartHolds] = useState(initialRouteState.startHolds)
  const [endHolds, setEndHolds] = useState(initialRouteState.endHolds)
  const [selectionMode, setSelectionMode] = useState('start')
  const [seed, setSeed] = useState(initialRouteState.seed)
  const [temperature, setTemperature] = useState(0.8)

  const [candidates, setCandidates] = useState([])
  const [selectedCandidate, setSelectedCandidate] = useState(0)
  const [isGenerating, setIsGenerating] = useState(false)
  const [generationMessage, setGenerationMessage] = useState('')
  const [error, setError] = useState('')
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    let active = true
    setServiceState('loading')
    setServiceMessage('Connecting to the route service…')
    const wakeTimer = window.setTimeout(() => {
      if (active) {
        setServiceMessage('Modal is waking the GPU. A first request can take about a minute.')
      }
    }, 8_000)

    getModelCapabilities()
      .then((model) => {
        if (!active) return
        const endpointHolds = normalizeEndpointHolds(model.endpoint_holds, model.board_holds)
        const normalizedModel = {
          ...model,
          board_holds: physicalBoardHolds(model.board_holds),
          endpoint_holds: endpointHolds,
        }
        setCapabilities(normalizedModel)
        setServiceState('ready')
        setServiceMessage('Model ready')

        if (!initialized.current) {
          const startsCatalog = new Set(endpointHolds.start)
          const endsCatalog = new Set(endpointHolds.end)
          const validStarts = initialRouteState.startHolds
            .filter((hold) => startsCatalog.has(hold))
            .slice(0, 2)
          const starts = validStarts.length
            ? validStarts
            : [startsCatalog.has('F4') ? 'F4' : endpointHolds.start[0]]
          const validEnds = initialRouteState.endHolds
            .filter((hold) => endsCatalog.has(hold) && !starts.includes(hold))
            .slice(0, 2)
          const fallbackEnd = endsCatalog.has('G17') && !starts.includes('G17')
            ? 'G17'
            : endpointHolds.end.find((hold) => !starts.includes(hold))

          setStartHolds(starts)
          setEndHolds(validEnds.length ? validEnds : [fallbackEnd])
          setSelectedGrade(
            model.supported_grades.includes(initialRouteState.grade)
              ? initialRouteState.grade
              : model.supported_grades.includes('7A')
                ? '7A'
                : model.supported_grades[0],
          )
          initialized.current = true
        }
      })
      .catch((requestError) => {
        if (!active) return
        setServiceState('error')
        setServiceMessage(getApiErrorMessage(requestError))
      })
      .finally(() => window.clearTimeout(wakeTimer))

    return () => {
      active = false
      window.clearTimeout(wakeTimer)
    }
  }, [capabilityAttempt])

  useEffect(() => {
    if (!capabilities || !selectedGrade) return
    const query = writeRouteState({ grade: selectedGrade, startHolds, endHolds, seed })
    const nextUrl = `${window.location.pathname}${query ? `?${query}` : ''}`
    window.history.replaceState(null, '', nextUrl)
  }, [capabilities, selectedGrade, startHolds, endHolds, seed])

  const clearResults = () => {
    setCandidates([])
    setSelectedCandidate(0)
    setGenerationMessage('')
    setError('')
  }

  const updateStarts = (holds) => {
    setStartHolds(holds)
    clearResults()
  }

  const updateEnds = (holds) => {
    setEndHolds(holds)
    clearResults()
  }

  const handleBoardHold = (hold) => {
    const allowedHolds = capabilities.endpoint_holds[selectionMode]
    if (!allowedHolds.includes(hold)) {
      setError(`${displayHoldLabel(hold)} is not available as a ${selectionMode} hold.`)
      return
    }
    const activeHolds = selectionMode === 'start' ? startHolds : endHolds
    const otherHolds = selectionMode === 'start' ? endHolds : startHolds
    if (otherHolds.includes(hold)) {
      setError(`${displayHoldLabel(hold)} is already selected as ${selectionMode === 'start' ? 'a finish' : 'a start'} hold.`)
      return
    }
    const next = activeHolds.includes(hold)
      ? activeHolds.filter((item) => item !== hold)
      : activeHolds.length < 2
        ? [...activeHolds, hold]
        : [activeHolds[0], hold]
    if (selectionMode === 'start') updateStarts(next)
    else updateEnds(next)
  }

  const applyPreset = (preset) => {
    if (!capabilities.supported_grades.includes(preset.grade)) return
    if (!preset.startHolds.every((hold) => capabilities.endpoint_holds.start.includes(hold))) return
    if (!preset.endHolds.every((hold) => capabilities.endpoint_holds.end.includes(hold))) return
    setSelectedGrade(preset.grade)
    setStartHolds(preset.startHolds)
    setEndHolds(preset.endHolds)
    clearResults()
  }

  const handleGenerate = async () => {
    setIsGenerating(true)
    setError('')
    setGenerationMessage('Sampling and checking route candidates…')
    const wakeTimer = window.setTimeout(() => {
      setGenerationMessage('The GPU is waking up. Keep this tab open—cold starts can take about a minute.')
    }, 8_000)

    try {
      const result = await generateRoutes({
        grade: selectedGrade,
        start_holds: startHolds,
        end_holds: endHolds,
        candidate_count: 3,
        temperature,
        top_p: 0.95,
        ...(seed === null ? {} : { seed }),
      })
      setCandidates(result.candidates)
      setSelectedCandidate(0)
      setSeed(result.seed)
      setGenerationMessage(
        `${result.candidates.length} valid routes · ${result.rejected_attempts} sampled routes filtered`,
      )
    } catch (requestError) {
      setError(getApiErrorMessage(requestError))
      setGenerationMessage('')
    } finally {
      window.clearTimeout(wakeTimer)
      setIsGenerating(false)
    }
  }

  const handleShare = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 2_000)
    } catch {
      setError('Copy was blocked by the browser. You can copy the current URL manually.')
    }
  }

  const activeCandidate = candidates[selectedCandidate] || null

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="/" aria-label="Route Forge home">
          <span className="brand-mark"><Mountain size={22} aria-hidden="true" /></span>
          <span><strong>Route Forge</strong><small>MoonBoard path studio</small></span>
        </a>
        <div className={`service-pill ${serviceState}`} role="status">
          <Activity size={15} aria-hidden="true" />
          {serviceState === 'ready' && capabilities
            ? `${capabilities.model_version} · ready`
            : serviceState === 'error' ? 'Service unavailable' : 'Connecting'}
        </div>
      </header>

      <section className="hero">
        <div>
          <span className="eyebrow hero-kicker">AI-assisted setting, human-reviewed movement</span>
          <h1>Build a problem.<br /><em>Read the movement.</em></h1>
        </div>
        <p>
          Choose the anchors. Route Forge returns one conservative match and two
          model alternatives, each checked against the MoonBoard layout.
        </p>
      </section>

      {serviceState !== 'ready' ? (
        <main className="connection-stage">
          <div className={`connection-card ${serviceState}`}>
            <span className="connection-orbit" aria-hidden="true" />
            <h2>{serviceState === 'error' ? 'Could not reach the wall' : 'Warming the wall'}</h2>
            <p>{serviceMessage}</p>
            {serviceState === 'error' && (
              <button type="button" onClick={() => setCapabilityAttempt((value) => value + 1)}>
                <RefreshCw size={17} aria-hidden="true" /> Try again
              </button>
            )}
          </div>
        </main>
      ) : (
        <main className="workspace">
          <aside className="workspace-controls">
            <Controls
              capabilities={capabilities}
              selectedGrade={selectedGrade}
              onGradeChange={(grade) => { setSelectedGrade(grade); clearResults() }}
              startHolds={startHolds}
              endHolds={endHolds}
              onStartChange={updateStarts}
              onEndChange={updateEnds}
              selectionMode={selectionMode}
              onSelectionModeChange={setSelectionMode}
              seed={seed}
              onSeedChange={setSeed}
              temperature={temperature}
              onTemperatureChange={setTemperature}
              onApplyPreset={applyPreset}
              onGenerate={handleGenerate}
              isGenerating={isGenerating}
            />
            {(generationMessage || error) && (
              <div className={error ? 'notice error' : 'notice'} role={error ? 'alert' : 'status'}>
                {error || generationMessage}
              </div>
            )}
            <CandidateList
              candidates={candidates}
              selectedIndex={selectedCandidate}
              onSelect={setSelectedCandidate}
            />
            {candidates.length > 0 && (
              <button className="share-button" type="button" onClick={handleShare}>
                {copied ? <Check size={17} aria-hidden="true" /> : <Copy size={17} aria-hidden="true" />}
                {copied ? 'Link copied' : 'Copy shareable setup'}
              </button>
            )}
          </aside>
          <div className="board-column">
            <MoonBoard
              boardHolds={capabilities.board_holds}
              candidate={activeCandidate}
              startHolds={startHolds}
              endHolds={endHolds}
              selectableHolds={capabilities.endpoint_holds[selectionMode]}
              selectionMode={selectionMode}
              onHoldSelect={handleBoardHold}
            />
          </div>
        </main>
      )}

      <footer>
        <span>Route Forge · Phase 4 preview</span>
        <span>Generated routes require human judgment.</span>
      </footer>
    </div>
  )
}

export default App
