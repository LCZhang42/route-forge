const CALIBRATION = {
  left: 14.3,
  right: 91.3,
  top: 8.9,
  bottom: 94.4,
}

export function parseHoldLabel(label) {
  return {
    x: label.charCodeAt(0) - 65,
    y: Number(label.slice(1)),
  }
}

export function holdPosition(label) {
  const { x, y } = parseHoldLabel(label)
  return {
    left: CALIBRATION.left + (x / 10) * (CALIBRATION.right - CALIBRATION.left),
    top: CALIBRATION.bottom - ((y - 1) / 17) * (CALIBRATION.bottom - CALIBRATION.top),
  }
}
