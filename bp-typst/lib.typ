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

#let csym(content) = circle(
  fill: black,
  radius: 0.5em,
  align(center + horizon, text(font: "DejaVu Sans Mono", fill: white, size: 0.65em, weight: "bold", content)),
)

#let annot-state = state("raw-annots", ())

#let raw-annot(..annots) = {
  annot-state.update(s => s + annots.pos())
  for annot in annots.pos() {
    if annot.at("label", default: none) != none {
      [#metadata(annot) #annot.label]
    }
  }
}

#let init-raw-annot(body) = {
  show raw.where(block: true): it => context {
    let annots = annot-state.get()
    annot-state.update(())
    block(
      inset: 8pt,
      width: 100%,
      stroke: (left: 1.5pt + luma(40%)),
      // this stack contains all the lines from top to bottom
      stack(
        dir: ttb,
        spacing: 0.65em,
        // put all the lines inside
        ..it.lines.map(line => {
          // find the annotation inside the state that is on the line number specified
          let annot = annots.find(a => a.line == line.number)
          stack(
            dir: ltr,
            spacing: 0.7em,
            line.body,
            if annot != none {
              box(
                width: 1em,
                height: 0pt,
                place(horizon, dy: 0.45em, text(size: 1em / 0.8, csym(annot.symb))),
              )
            },
          )
        }),
      ),
    )
  }

  show ref: it => context {
    let el = it.element
    if el != none and el.func() == metadata {
      link(el.location(), box(
        width: 1em, // reserve horizontal space so it doesn't overlap text
        height: 0pt, // contribute zero height to the line
        place(horizon, dy: -0.4em, csym(el.value.symb)),
      ))
    } else { it }
  }

  // el.value.symb

  body // <-- the rest of the document goes here
}
