#let template(body) = {
  // Same font as the BUT template
  set text(font: "New Computer Modern", lang: "en")

  // idk what this does
  show math.equation: set text(weight: 400)

  // first paragraph isn't indented, the rest is
  // reduces the spacing between paragraphs (to it's similar to the BUT template)
  set par(justify: true, first-line-indent: 2em, spacing: 0.7em)

  // figures are numbered by the chapter they're in
  set figure(numbering: num => {
    let chapter = counter(heading).get().at(0, default: 0)
    (chapter, num).map(str).join(".")
  })

  // figures have a little bit more space around them
  show figure: set block(above: 2em, below: 2em)

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

  // all chapters start on new page and have a nice chapter heading
  show heading.where(depth: 1): it => {
    if it.numbering == none { it } else {
      pagebreak()
      let chapter-no = numbering("1", ..counter(heading).at(it.location()))
      v(3em)
      block(below: 2em)[
        #text(weight: "bold", size: 1.2em)[Chapter #chapter-no]
        // FIXME: this is disgusting and the person who wrote it should be ashamed of themselves
        #heading(numbering: none, outlined: false)[#it.body]
      ]
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
    #heading(outlined: false, level: 2)[Abstract]

    #abstract-en
  ],
  [],
  [
    #heading(outlined: false, level: 2)[Abstrakt]
    #show: slovak-text

    #abstract-sk
  ],
  [],
  [
    #heading(outlined: false, level: 2)[Keywords]

    #keywords-en
  ],
  [],
  [
    #heading(outlined: false, level: 2)[Kľúčové slová]
    #show: slovak-text

    #keywords-sk
  ],
  [],
  [
    #heading(outlined: false, level: 2)[Reference]

    #reference
  ],
)
