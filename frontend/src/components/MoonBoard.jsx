import { useEffect, useRef, useState } from 'react'

const GRID_WIDTH = 11
const GRID_HEIGHT = 18

function MoonBoard({ path, grade }) {
  const canvasRef = useRef(null)
  const containerRef = useRef(null)
  const [dimensions, setDimensions] = useState({ width: 600, height: 980 })
  const [backgroundLoaded, setBackgroundLoaded] = useState(false)
  const backgroundImageRef = useRef(null)

  useEffect(() => {
    const img = new Image()
    img.src = '/data/moonboard2016Background.jpg'
    img.onload = () => {
      backgroundImageRef.current = img
      setBackgroundLoaded(true)
      
      const aspectRatio = img.height / img.width
      const containerWidth = containerRef.current?.offsetWidth || 600
      const maxWidth = Math.min(containerWidth - 40, 700)
      
      setDimensions({
        width: maxWidth,
        height: maxWidth * aspectRatio
      })
    }
    img.onerror = () => {
      console.warn('Could not load moonboard background, using fallback')
      setBackgroundLoaded(false)
    }
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    const { width, height } = dimensions

    ctx.clearRect(0, 0, width, height)

    if (backgroundLoaded && backgroundImageRef.current) {
      ctx.drawImage(backgroundImageRef.current, 0, 0, width, height)
    } else {
      ctx.fillStyle = '#fbbf24'
      ctx.fillRect(0, 0, width, height)
    }

    const cellWidth = width / GRID_WIDTH
    const cellHeight = height / GRID_HEIGHT

    const coordinateToPixel = (x, y) => {
      const pixelX = (x + 0.5) * cellWidth
      const pixelY = (GRID_HEIGHT - y + 0.5) * cellHeight
      return [pixelX, pixelY]
    }

    if (path && path.length > 0) {
      const drawDashedLine = (x1, y1, x2, y2, color, thickness) => {
        const dx = x2 - x1
        const dy = y2 - y1
        const length = Math.sqrt(dx * dx + dy * dy)
        const dashLength = 15
        const numDashes = Math.floor(length / dashLength)
        
        ctx.strokeStyle = color
        ctx.lineWidth = thickness
        ctx.lineCap = 'round'
        
        for (let i = 0; i < numDashes; i += 2) {
          const t1 = i / numDashes
          const t2 = Math.min((i + 1) / numDashes, 1)
          
          ctx.beginPath()
          ctx.moveTo(x1 + dx * t1, y1 + dy * t1)
          ctx.lineTo(x1 + dx * t2, y1 + dy * t2)
          ctx.stroke()
        }
      }

      for (let i = 0; i < path.length - 1; i++) {
        const [x1, y1] = path[i]
        const [x2, y2] = path[i + 1]
        
        const [px1, py1] = coordinateToPixel(x1, y1)
        const [px2, py2] = coordinateToPixel(x2, y2)
        
        drawDashedLine(px1, py1, px2, py2, '#fcd34d', 5)
      }

      path.forEach(([x, y], index) => {
        const [centerX, centerY] = coordinateToPixel(x, y)

        let color
        if (index === 0) {
          color = '#10b981'
        } else if (index === path.length - 1) {
          color = '#ef4444'
        } else {
          color = '#3b82f6'
        }

        const radius = Math.min(cellWidth, cellHeight) * 0.35

        ctx.globalAlpha = 0.3
        ctx.fillStyle = color
        ctx.beginPath()
        ctx.arc(centerX, centerY, radius - 3, 0, Math.PI * 2)
        ctx.fill()
        ctx.globalAlpha = 1.0

        ctx.strokeStyle = color
        ctx.lineWidth = 5
        ctx.beginPath()
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2)
        ctx.stroke()

        ctx.strokeStyle = '#ffffff'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2)
        ctx.stroke()

        ctx.fillStyle = '#ffffff'
        ctx.font = `bold ${Math.floor(radius * 0.8)}px sans-serif`
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.shadowColor = 'rgba(0, 0, 0, 0.8)'
        ctx.shadowBlur = 4
        ctx.fillText(index + 1, centerX, centerY)
        ctx.shadowBlur = 0
      })
    }
  }, [path, dimensions, backgroundLoaded])

  return (
    <div className="bg-slate-800 rounded-lg p-6 shadow-2xl" ref={containerRef}>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold text-white">MoonBoard 2016</h2>
        {grade && (
          <span className="px-4 py-2 bg-blue-500 text-white rounded-lg font-bold">
            Grade: {grade}
          </span>
        )}
      </div>
      <div className="flex justify-center bg-slate-900 rounded-lg p-4">
        <canvas
          ref={canvasRef}
          width={dimensions.width}
          height={dimensions.height}
          className="border-2 border-slate-700 rounded max-w-full"
        />
      </div>
      {path && path.length > 0 && (
        <div className="mt-4 text-sm text-slate-300">
          <div className="flex items-center justify-center gap-6 mb-3">
            <span className="inline-flex items-center gap-2">
              <span className="w-3 h-3 bg-green-500 rounded-full"></span> Start
            </span>
            <span className="inline-flex items-center gap-2">
              <span className="w-3 h-3 bg-blue-500 rounded-full"></span> Mid
            </span>
            <span className="inline-flex items-center gap-2">
              <span className="w-3 h-3 bg-red-500 rounded-full"></span> End
            </span>
          </div>
          <div className="text-center text-slate-400">
            {path.length} holds total
          </div>
        </div>
      )}
    </div>
  )
}

export default MoonBoard
