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

#include "ml/index.typ"
#pagebreak()
#include "math/index.typ"
#pagebreak()
#include "stats/index.typ"
#pagebreak()
#include "neuro/index.typ"

#pagebreak()
#bibliography("ref.bib", style: "ieee", title: "参考文献")

#pagebreak()
#columns(2)[
  #make-index(title: [Index], outlined: true)
]
