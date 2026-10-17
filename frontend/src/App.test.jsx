import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
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
  board_holds: ['A4', 'D8', 'E10', 'F12', 'G17'],
  endpoint_holds: { start: ['A4'], end: ['G17'] },
  generation_modes: [
    { id: 'safer', label: 'Safer', description: 'Common reach for this grade' },
    { id: 'balanced', label: 'Balanced', description: 'Moderate reach variation' },
    { id: 'dynamic', label: 'Dynamic', description: 'Rare, powerful moves allowed' },
  ],
  profiles: { '7A': { soft_move_limit: 5 } },
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
    warnings: ['one or more moves are unusually long for this grade'],
    move_distances: [5, 4.472, 5.099],
    maximum_move_distance: 5.099,
    novel_route: false,
    generation_mode: 'balanced',
    feasibility_limit: 5.75,
    plausible_path: ['A4', 'D8', 'F12', 'G17'],
  },
}

describe('App Phase 4 contract', () => {
  afterEach(cleanup)

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
    expect(screen.getByText('Quick setup')).toBeInTheDocument()
    expect(screen.getByText('Quick setup')).toHaveAccessibleName(
      /Optional examples for 6C\+, 7A, and 7B/i,
    )
    expect(screen.getByRole('button', { name: /Two-hand: Preset/i })).toBeInTheDocument()
    expect(screen.queryByText(/Allows modest reach outliers/i)).not.toBeInTheDocument()
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
      generation_mode: 'balanced',
    })))
    expect(await screen.findByText('Conservative match')).toBeInTheDocument()
    expect(screen.getByText('Large move · F13 → G18 · 5.1 grid units')).toBeInTheDocument()
    expect(screen.getByText('Longer than 95% of recorded 7A moves')).toBeInTheDocument()
    expect(screen.queryByText(/3 balanced routes/i)).not.toBeInTheDocument()
    expect(screen.getByText(/1 balanced route/i)).toBeInTheDocument()
  })

  it('edits a route on the board and restores the generated holds', async () => {
    render(<App />)
    await userEvent.click(await screen.findByRole('button', { name: 'Forge routes' }))
    await screen.findByText('Conservative match')

    await userEvent.click(screen.getByRole('button', { name: 'Edit route' }))
    await userEvent.click(screen.getByRole('button', { name: /D9, route hold.*Select this ring to move/i }))
    await userEvent.click(screen.getByRole('button', { name: /E11.*Move D9 here/i }))

    expect(screen.getByRole('button', { name: /E11, route hold/i })).toHaveClass('is-route')
    expect(screen.getByText('Custom route')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Restore generated route' }))
    expect(screen.getByRole('button', { name: /D9, route hold/i })).toHaveClass('is-route')
    expect(screen.queryByText('Custom route')).not.toBeInTheDocument()
  })
})
