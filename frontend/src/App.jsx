import { useState } from 'react'
import MoonBoard from './components/MoonBoard'
import GradeSelector from './components/GradeSelector'
import Controls from './components/Controls'
import { Mountain } from 'lucide-react'

function App() {
  const [selectedGrade, setSelectedGrade] = useState('7A')
  const [generatedPath, setGeneratedPath] = useState(null)
  const [isGenerating, setIsGenerating] = useState(false)

  const handleGenerate = async (options) => {
    setIsGenerating(true)
    
    try {
      const { generatePath } = await import('./api/client')
      const result = await generatePath(selectedGrade, options)
      setGeneratedPath(result)
    } catch (error) {
      console.error('Failed to generate path:', error)
      alert('Failed to generate path. Make sure the backend server is running.')
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Mountain className="w-12 h-12 text-blue-400" />
            <h1 className="text-5xl font-bold text-white">Climb Path Generator</h1>
          </div>
          <p className="text-slate-300 text-lg">
            AI-powered MoonBoard route generation
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-1 space-y-6">
            <GradeSelector 
              selectedGrade={selectedGrade}
              onGradeChange={setSelectedGrade}
            />
            <Controls 
              onGenerate={handleGenerate}
              isGenerating={isGenerating}
              selectedGrade={selectedGrade}
            />
          </div>

          <div className="lg:col-span-2">
            <MoonBoard 
              path={generatedPath?.path}
              grade={generatedPath?.grade}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
