import { useState } from 'react'
import { Sparkles, Settings } from 'lucide-react'

function Controls({ onGenerate, isGenerating, selectedGrade }) {
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [temperature, setTemperature] = useState(1.0)
  const [minHolds, setMinHolds] = useState(3)
  const [maxHolds, setMaxHolds] = useState(30)

  const handleGenerate = () => {
    const options = {
      temperature: parseFloat(temperature),
      min_holds: parseInt(minHolds),
      max_holds: parseInt(maxHolds),
      use_constraints: true,
    }
    
    onGenerate(options)
  }

  return (
    <div className="bg-slate-800 rounded-lg p-6 shadow-2xl">
      <h3 className="text-xl font-bold text-white mb-4">Generation Controls</h3>
      
      <button
        onClick={handleGenerate}
        disabled={isGenerating}
        className={`w-full py-4 rounded-lg font-bold text-lg flex items-center justify-center gap-2 transition-all ${
          isGenerating
            ? 'bg-slate-600 text-slate-400 cursor-not-allowed'
            : 'bg-gradient-to-r from-blue-500 to-purple-600 text-white hover:from-blue-600 hover:to-purple-700 shadow-lg hover:shadow-xl'
        }`}
      >
        <Sparkles className="w-5 h-5" />
        {isGenerating ? 'Generating...' : 'Generate Path'}
      </button>

      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="w-full mt-3 py-2 bg-slate-700 text-slate-300 rounded-lg hover:bg-slate-600 transition-all flex items-center justify-center gap-2"
      >
        <Settings className="w-4 h-4" />
        {showAdvanced ? 'Hide' : 'Show'} Advanced Options
      </button>

      {showAdvanced && (
        <div className="mt-4 space-y-4">
          <div className="p-4 bg-slate-700 rounded-lg">
            <h4 className="text-sm font-semibold text-white mb-2">Temperature</h4>
            <div className="flex items-center gap-3">
              <input
                type="range"
                value={temperature}
                onChange={(e) => setTemperature(e.target.value)}
                min="0.1"
                max="2.0"
                step="0.1"
                className="flex-1"
              />
              <span className="text-white font-mono w-12 text-right">{temperature}</span>
            </div>
            <p className="text-xs text-slate-400 mt-1">Higher = more random paths</p>
          </div>

          <div className="p-4 bg-slate-700 rounded-lg">
            <h4 className="text-sm font-semibold text-white mb-2">Hold Count Range</h4>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-xs text-slate-400">Min</label>
                <input
                  type="number"
                  value={minHolds}
                  onChange={(e) => setMinHolds(e.target.value)}
                  min="1"
                  max="50"
                  className="w-full px-3 py-2 bg-slate-600 text-white rounded border border-slate-500 focus:border-blue-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="text-xs text-slate-400">Max</label>
                <input
                  type="number"
                  value={maxHolds}
                  onChange={(e) => setMaxHolds(e.target.value)}
                  min="1"
                  max="50"
                  className="w-full px-3 py-2 bg-slate-600 text-white rounded border border-slate-500 focus:border-blue-500 focus:outline-none"
                />
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="mt-4 p-3 bg-slate-700 rounded-lg">
        <p className="text-xs text-slate-400">
          Generate a climbing path for grade <span className="font-semibold text-white">{selectedGrade}</span>
        </p>
      </div>
    </div>
  )
}

export default Controls
