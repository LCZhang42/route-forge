import { describe, expect, it } from 'vitest'
import { readRouteState, writeRouteState } from './state'

describe('shareable route state', () => {
  it('round-trips non-secret route inputs', () => {
    const query = writeRouteState({
      grade: '7A',
      startHolds: ['A4', 'F4'],
      endHolds: ['G17'],
      seed: 42,
    })

    expect(readRouteState(`?${query}`)).toEqual({
      grade: '7A',
      startHolds: ['A4', 'F4'],
      endHolds: ['G17'],
      seed: 42,
    })
  })

  it('drops invalid seed values and limits endpoint cardinality', () => {
    expect(readRouteState('?start=A4,F4,D5&end=G17&seed=nope')).toEqual({
      grade: null,
      startHolds: ['A4', 'F4'],
      endHolds: ['G17'],
      seed: null,
    })
  })
})
