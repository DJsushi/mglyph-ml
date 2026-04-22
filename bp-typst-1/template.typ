#let template(body) = {
  set text(font: "New Computer Modern", lang: "en")
  show math.equation: set text(weight: 400) // idk what this does
  set par(justify: true)

  // Match the LaTeX thesis template layout on A4 as closely as possible.
  set page(
    paper: "a4",
    margin: (
      left: 3.4cm,
      right: 2.4cm,
      top: 3.7cm,
      bottom: 3.7cm,
    ),
  )

  show heading: set text(size: 1.2em)

  // all top-level chapters start on new page
  show heading.where(depth: 1): it => {
    if it.numbering == none {
      it
    } else {
      pagebreak()
      let chapter-no = numbering("1", ..counter(heading).at(it.location()))
      block(below: 0.8em)[
        #text(weight: "bold")[Chapter #chapter-no]
      ]
      it
    }
  }

  // do not break lines in inline equations
  show math.equation.where(block: false): box

  // show links as blue
  show link: set text(rgb("#3366CC"))

  body
}

#let slovak-text(body) = {
  set text(lang: "sk")

  show " k ": [ k~]
  show " s ": [ s~]
  show " v ": [ v~]
  show " z ": [ z~]
  show " o ": [ o~]
  show " u ": [ u~]
  show " a ": [ a~]
  show " i ": [ i~]

  body
}

#let TODO(body) = text(body, weight: "bold", fill: red)

// Match LaTeX's `\vfill` behavior: content keeps natural height,
// and only inter-section gaps flex.
#let abstract-keywords(abstract-en: [], abstract-sk: [], keywords-en: [], keywords-sk: [], reference: []) = grid(
  rows: (auto, 1fr, auto, 1fr, auto, 1fr, auto, 1fr, auto),
  [
    == Abstract

    #abstract-en
  ],
  [],
  [
    == Abstrakt
    #show: slovak-text

    #abstract-sk
  ],
  [],
  [
    == Keywords

    #keywords-en
  ],
  [],
  [
    == Kľúčové slová
    #show: slovak-text

    #keywords-sk
  ],
  [],
  [
    == Reference

    #reference
  ],
)
