import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App'
import { generateRoutes, getModelCapabilities } from './api/client'

vi.mock('./api/client', () => ({
  getModelCapabilities: vi.fn(),
  generateRoutes: vi.fn(),
  getApiErrorMessage: vi.fn(() => 'Service unavailable'),
}))

const capabilities = {
  model_version: 'structured-v1',
  data_version: 'data-v1',
  supported_grades: ['7A'],
  board_holds: ['A4', 'D8', 'F12', 'G17'],
}

const candidate = {
  rank: 1,
  source: 'retrieval_fallback',
  grade: '7A',
  start_holds: ['A4'],
  middle_holds: ['D8', 'F12'],
  end_holds: ['G17'],
  holds: ['A4', 'D8', 'F12', 'G17'],
  diagnostics: {
    warnings: [],
    maximum_move_distance: 4.5,
    novel_route: false,
  },
}

describe('App Phase 4 contract', () => {
  beforeEach(() => {
    window.history.replaceState(null, '', '/')
    getModelCapabilities.mockResolvedValue(capabilities)
    generateRoutes.mockResolvedValue({
      seed: 42,
      rejected_attempts: 2,
      candidates: [candidate],
    })
  })

  it('loads model capabilities instead of hard-coding grades', async () => {
    render(<App />)

    expect(await screen.findByRole('heading', { name: 'Movement canvas' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: '7A' })).toHaveAttribute('aria-checked', 'true')
    expect(getModelCapabilities).toHaveBeenCalled()
  })

  it('sends the Phase 3 request contract and renders diagnostics', async () => {
    render(<App />)
    const generate = await screen.findByRole('button', { name: 'Forge routes' })
    await userEvent.click(generate)

    await waitFor(() => expect(generateRoutes).toHaveBeenCalledWith(expect.objectContaining({
      grade: '7A',
      start_holds: ['A4'],
      end_holds: ['G17'],
      candidate_count: 3,
    })))
    expect(await screen.findByText('Conservative match')).toBeInTheDocument()
    expect(screen.queryByText(/3 valid routes/i)).not.toBeInTheDocument()
    expect(screen.getByText(/1 valid routes/i)).toBeInTheDocument()
  })
})
