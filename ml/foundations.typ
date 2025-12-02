#import "../prelude.typ": *

#let ml_foundations(include_bibliography: true) = [
  #hd2("Foundations")

  #include "foundations/nfl.typ"
  #include "foundations/monte-carlo.typ"
  #include "foundations/cnn.typ"
  #include "foundations/state-space.typ"
  #include "foundations/fourier-transform.typ"

  #if include_bibliography {
    pagebreak()
    bibliography("/ref.bib", style: "ieee", title: "参考文献")
  }
]

#ml_foundations()
