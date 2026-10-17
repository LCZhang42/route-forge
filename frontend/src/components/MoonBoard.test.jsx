import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import MoonBoard from './MoonBoard'
import {
  displayHoldLabel,
  fallbackEndpointHolds,
  holdPosition,
  normalizeEndpointHolds,
  physicalBoardHolds,
} from './boardGeometry'

describe('MoonBoard', () => {
  it('maps labels to the calibrated image grid', () => {
    expect(holdPosition('A4').left).toBeCloseTo(14.3)
    expect(holdPosition('K17').left).toBeCloseTo(91.3)
    expect(holdPosition('K17').top).toBeCloseTo(8.9, 1)
    expect(displayHoldLabel('A4')).toBe('A5')
  })

  it('derives endpoint choices without the phantom physical A6 coordinate', () => {
    expect(fallbackEndpointHolds(['A4', 'A5', 'D8', 'G17'])).toEqual({
      start: ['A4'],
      end: ['G17'],
    })
    expect(physicalBoardHolds(['A4', 'A5', 'G17'])).toEqual(['A4', 'G17'])
    expect(normalizeEndpointHolds(
      { start: ['A4', 'A5'], end: ['G17'] },
      ['A4', 'A5', 'G17'],
    )).toEqual({ start: ['A4'], end: ['G17'] })
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
        onRouteHoldSelect={vi.fn()}
        onRouteHoldMove={vi.fn()}
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
        onRouteHoldSelect={vi.fn()}
        onRouteHoldMove={vi.fn()}
      />,
    )

    expect(screen.getByRole('button', { name: /B8, route hold/i })).toHaveClass('is-route')
    expect(container.querySelector('.route-lines')).not.toBeInTheDocument()
    expect(container.querySelector('.board-hold span')).not.toBeInTheDocument()
  })

  it('drags a route ring to a new hold and restores the generated route', async () => {
    const onRouteHoldSelect = vi.fn()
    const onRouteHoldMove = vi.fn()
    const onRouteRestore = vi.fn()
    render(
      <MoonBoard
        boardHolds={['A4', 'B7', 'C8', 'G17']}
        candidate={{
          holds: ['A4', 'B7', 'G17'],
          start_holds: ['A4'],
          middle_holds: ['B7'],
          end_holds: ['G17'],
        }}
        startHolds={['A4']}
        endHolds={['G17']}
        selectableHolds={['A4']}
        selectionMode="start"
        onHoldSelect={vi.fn()}
        routeEditMode
        routeModified
        movingHold="B7"
        onRouteEditToggle={vi.fn()}
        onRouteHoldSelect={onRouteHoldSelect}
        onRouteHoldMove={onRouteHoldMove}
        onRouteRestore={onRouteRestore}
      />,
    )

    const source = screen.getByRole('button', { name: /B8.*Drag this ring/i })
    const destination = screen.getByRole('button', { name: /C9.*Press Enter to move B8 here/i })
    const dataTransfer = {
      effectAllowed: '',
      dropEffect: '',
      getData: vi.fn(() => 'B7'),
      setData: vi.fn(),
    }

    fireEvent.dragStart(source, { dataTransfer })
    fireEvent.dragOver(destination, { dataTransfer })
    fireEvent.drop(destination, { dataTransfer })
    expect(onRouteHoldMove).toHaveBeenCalledWith('B7', 'C8')

    await userEvent.click(screen.getByRole('button', { name: 'Restore generated route' }))
    expect(onRouteRestore).toHaveBeenCalled()
  })
})
