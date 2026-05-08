#let template(printing: false, body) = {
  let block-spacing = 2em

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
  show figure: set block(above: block-spacing, below: block-spacing)

  // increase a little bit the gap between the figure and the caption... it was too small IMO
  set figure(gap: 1.2em)

  // code blocks have a little bit more space around them
  show raw: set block(above: block-spacing, below: block-spacing)

  // don't break lines of code (gotta override this behavior for long code snippets but i don't think there are many, so that's why i am changing this default behavior)
  show raw: set block(breakable: false)

  // when citing using prose, don't write all authors... If more than 2, write et al.
  set cite(style: "res/iso690-numeric-en-fitvut-short.csl")

  // Match the LaTeX thesis template layout on A4 as closely as possible.
  set page(
    paper: "a4",
    margin: if printing {
      (
        inside: 3.4cm,
        outside: 2.4cm,
        y: 3.7cm,
      )
    } else {
      (
        left: 3.4cm,
        right: 2.4cm,
        y: 3.7cm,
      )
    },
  )

  show heading: set text(size: 1.2em)

  // we need numbered equations!
  set math.equation(numbering: "(1)")

  // also space around math equations
  show math.equation.where(block: true): set block(above: block-spacing, below: block-spacing)

  // all chapters start on new page and have a nice chapter heading
  show heading.where(depth: 1): it => {
    if it.numbering == none { it } else {
      pagebreak()
      let chapter-no = numbering("1", ..counter(heading).at(it.location()))
      v(3em)
      block(below: 2em)[
        #text(weight: "bold", size: 1.2em)[Chapter #chapter-no]
        // FIXME: this is disgusting and the person who wrote it should be ashamed of themselves
        #text(size: 0.85em)[
          #heading(numbering: none, outlined: false)[#it.body]
        ]
      ]
    }

    // we have to reset the figure counters each chapter because otherwise we get incorrect figure numbering
    counter(figure.where(kind: image)).update(0)
    counter(figure.where(kind: table)).update(0)
  }

  // do not break lines in inline equations
  show math.equation.where(block: false): box

  // show links as blue
  show link: set text(rgb("#3366CC"))

  // when referring to chapters, say "chapter" instead of "section"
  show heading.where(level: 1): set heading(supplement: [Chapter])

  // citations of authors by the CSN ISO 690 are all caps and have a dot at the end...
  show cite.where(form: "author"): set cite(style: "ieee")
  show cite.where(form: "prose"): set cite(style: "ieee")

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

#let teal-fill = teal.lighten(70%)
#let teal-stroke = teal.darken(10%)
#let purple-fill = purple.lighten(70%)
#let purple-stroke = purple.darken(10%)
#let green-fill = green.lighten(70%)
#let green-stroke = green.darken(10%)
