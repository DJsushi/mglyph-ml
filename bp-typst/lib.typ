#import "@preview/cetz:0.5.0": canvas, draw

#let number-line(points: (0, 25, 50, 75, 100), start: 0, end: 100, length: 10) = canvas({
  import draw: *

  let span = end - start
  let x-start = 0
  let x-end = length

  // Main line
  line((x-start, 0), (x-end, 0), mark: (end: ">", start: ">"))

  // Ticks and labels
  for p in points {
    let t = if span == 0 { 0 } else { (p - start) / span }
    let x = x-start + t * (x-end - x-start)
    line((x, -0.15), (x, 0.15))
    content((x, -0.5), [#p])
  }
})
