import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import MoonBoard from './MoonBoard'
import { displayHoldLabel, fallbackEndpointHolds, holdPosition } from './boardGeometry'

describe('MoonBoard', () => {
  it('maps labels to the calibrated image grid', () => {
    expect(holdPosition('A4').left).toBeCloseTo(14.3)
    expect(holdPosition('K17').left).toBeCloseTo(91.3)
    expect(holdPosition('K17').top).toBeCloseTo(8.9, 1)
    expect(displayHoldLabel('A4')).toBe('A5')
  })

  it('derives endpoint choices from observed lower holds and the finish row', () => {
    expect(fallbackEndpointHolds(['A4', 'D8', 'G17'])).toEqual({
      start: ['A4'],
      end: ['G17'],
    })
  })

  it('exposes every available hold as a keyboard-accessible button', async () => {
    const onHoldSelect = vi.fn()
    render(
      <MoonBoard
        boardHolds={['A4', 'G17']}
        candidate={null}
        startHolds={['A4']}
        endHolds={['G17']}
        selectableHolds={['A4', 'G17']}
        selectionMode="start"
        onHoldSelect={onHoldSelect}
      />,
    )

    await userEvent.click(screen.getByRole('button', { name: /G18.*Set as start/i }))
    expect(onHoldSelect).toHaveBeenCalledWith('G17')
    expect(screen.getByRole('button', { name: /A5, selected start/i })).toHaveAttribute('aria-pressed', 'true')
  })

  it('marks route holds with rings without prescribing a numbered sequence', () => {
    const candidate = {
      holds: ['A4', 'B7', 'G17'],
      start_holds: ['A4'],
      middle_holds: ['B7'],
      end_holds: ['G17'],
    }
    const { container } = render(
      <MoonBoard
        boardHolds={candidate.holds}
        candidate={candidate}
        startHolds={['A4']}
        endHolds={['G17']}
        selectableHolds={candidate.holds}
        selectionMode="start"
        onHoldSelect={vi.fn()}
      />,
    )

    expect(screen.getByRole('button', { name: /B8, route hold/i })).toHaveClass('is-route')
    expect(container.querySelector('.route-lines')).not.toBeInTheDocument()
    expect(container.querySelector('.board-hold span')).not.toBeInTheDocument()
  })
})
