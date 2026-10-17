import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import MoonBoard from './MoonBoard'
import { holdPosition } from './boardGeometry'

describe('MoonBoard', () => {
  it('maps labels to the calibrated image grid', () => {
    expect(holdPosition('A4').left).toBeCloseTo(14.3)
    expect(holdPosition('K17').left).toBeCloseTo(91.3)
    expect(holdPosition('K17').top).toBeCloseTo(13.93, 1)
  })

  it('exposes every available hold as a keyboard-accessible button', async () => {
    const onHoldSelect = vi.fn()
    render(
      <MoonBoard
        boardHolds={['A4', 'G17']}
        candidate={null}
        startHolds={['A4']}
        endHolds={['G17']}
        selectionMode="start"
        onHoldSelect={onHoldSelect}
      />,
    )

    await userEvent.click(screen.getByRole('button', { name: /G17.*Set as start/i }))
    expect(onHoldSelect).toHaveBeenCalledWith('G17')
    expect(screen.getByRole('button', { name: /A4, selected start/i })).toHaveAttribute('aria-pressed', 'true')
  })
})
