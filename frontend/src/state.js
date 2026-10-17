const splitHolds = (value) =>
  value
    ?.split(',')
    .map((hold) => hold.trim().toUpperCase())
    .filter(Boolean)
    .slice(0, 2) ?? []

export function readRouteState(search = '') {
  const params = new URLSearchParams(search)
  const seedValue = params.get('seed')
  const parsedSeed = seedValue === null ? null : Number(seedValue)
  return {
    grade: params.get('grade') || null,
    startHolds: splitHolds(params.get('start')),
    endHolds: splitHolds(params.get('end')),
    seed: Number.isInteger(parsedSeed) && parsedSeed >= 0 ? parsedSeed : null,
  }
}

export function writeRouteState({ grade, startHolds, endHolds, seed }) {
  const params = new URLSearchParams()
  if (grade) params.set('grade', grade)
  if (startHolds?.length) params.set('start', startHolds.join(','))
  if (endHolds?.length) params.set('end', endHolds.join(','))
  if (Number.isInteger(seed) && seed >= 0) params.set('seed', String(seed))
  return params.toString()
}
