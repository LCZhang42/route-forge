const CALIBRATION = {
  left: 14.3,
  right: 91.3,
  top: 8.9,
  bottom: 94.4,
}

// The source dataset was extracted from the MoonBoard LED overlay. Its row
// values identify the LED below a hold, so the physical hold is one printed
// board row higher. Keep the legacy values at the API boundary until the model
// is retrained, but always present and draw the official physical hold label.
export const DISPLAY_ROW_OFFSET = 1

// The legacy export contains one phantom coordinate: internal A5, displayed as
// physical A6. It occurs in a single malformed three-start route, but the 2016
// setup has no handhold at that grid position. Keep the model token for
// checkpoint compatibility and exclude it at the UI/API boundary.
export const NON_PHYSICAL_2016_HOLDS = new Set(['A5'])

export function parseHoldLabel(label) {
  return {
    x: label.charCodeAt(0) - 65,
    y: Number(label.slice(1)),
  }
}

export function holdPosition(label) {
  const { x, y } = parseHoldLabel(label)
  const physicalRow = y + DISPLAY_ROW_OFFSET
  return {
    left: CALIBRATION.left + (x / 10) * (CALIBRATION.right - CALIBRATION.left),
    top: CALIBRATION.bottom - ((physicalRow - 1) / 17) * (CALIBRATION.bottom - CALIBRATION.top),
  }
}

export function displayHoldLabel(label) {
  const { x, y } = parseHoldLabel(label)
  return `${String.fromCharCode(65 + x)}${y + DISPLAY_ROW_OFFSET}`
}

export function physicalBoardHolds(boardHolds) {
  return boardHolds.filter((hold) => !NON_PHYSICAL_2016_HOLDS.has(hold))
}

export function fallbackEndpointHolds(boardHolds) {
  const physicalHolds = physicalBoardHolds(boardHolds)
  return {
    start: physicalHolds.filter((hold) => parseHoldLabel(hold).y <= 5),
    end: physicalHolds.filter((hold) => parseHoldLabel(hold).y === 17),
  }
}


export function normalizeEndpointHolds(endpointHolds, boardHolds) {
  const fallback = fallbackEndpointHolds(boardHolds)
  const advertised = endpointHolds || fallback
  return {
    start: physicalBoardHolds(advertised.start || fallback.start),
    end: physicalBoardHolds(advertised.end || fallback.end),
  }
}
