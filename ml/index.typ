#import "../prelude.typ": *

#import "foundations.typ": ml_foundations

#let ml_index(include_bibliography: true) = [
  #hd1("Machine Learning")
  #pagebreak()

  #ml_foundations(include_bibliography: false)
  #include "bayesian.typ"
  #include "latent-variable.typ"
  #include "graphical-models.typ"
  #include "optimization.typ"
  #include "clustering.typ"
  #include "long-sequence.typ"
  #include "misc-algorithms.typ"

  #if include_bibliography {
    pagebreak()
    bibliography("/ref.bib", style: "ieee", title: "参考文献")
  }
]

#ml_index()
