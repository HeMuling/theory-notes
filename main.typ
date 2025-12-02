#import "prelude.typ": *

#align(horizon + center)[#text(size: 28pt)[理论笔记]]

#pagebreak()

#outline(
  indent: 2em,
  depth: 3,
  title: "大纲",
)

#pagebreak()

#outline(
  indent: 2em,
  depth: 4,
  title: "目录",
)

#pagebreak()

#import "ml/index.typ": ml_index

#show bibliography: it => none

#ml_index(include_bibliography: false)
#pagebreak()
#include "math/index.typ"
#pagebreak()
#include "stats/index.typ"
#pagebreak()
#include "neuro/index.typ"

#pagebreak()
#show bibliography: it => it
#bibliography("ref.bib", style: "ieee", title: "参考文献")

#pagebreak()
#columns(2)[
  #make-index(title: [Index], outlined: true)
]
