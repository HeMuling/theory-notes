// 全局导入与样式设置
#import "@preview/drafting:0.2.2": margin-note, set-page-properties
#import "@preview/showybox:2.0.4": showybox
#import "@preview/lovelace:0.3.0": *
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#import "@preview/in-dexter:0.7.0": *

#set text(lang: "cn", font: "Source Han Sans SC")

#let include_bibliography() = sys.inputs.at("include-bibliography", default: true)

#set page(
  paper: "a4",
  numbering: "1",
  margin: (x: 1.6in),
)

#set par(
  justify: true,
  first-line-indent: 0em,
)

#set-page-properties()

// 标题工具（沿用旧版样式，便于逐步迁移）
#let hd1(in_text) = {
  text(size: 18pt)[
    #align(horizon)[
      #heading(level: 1)[#in_text]
      #v(10pt)
    ]
  ]
}
#let hd2(in_text) = {
  text(size: 16pt)[
    #align(center)[
      #heading(level: 2)[#in_text]
      #v(5pt)
    ]
  ]
}
#let hd3(in_text) = {
  text(size: 14pt)[
    #align(center)[
      #heading(level: 3)[#in_text]
    ]
  ]
}
#let hd4(in_text) = {
  text(size: 12pt)[
    #align(left)[
      #heading(level: 4)[#in_text]
    ]
  ]
}

#set heading(numbering: (..numbers) => {
  let level = numbers.pos().len()
  if level == 2 {
    return numbering("第一章", numbers.pos().at(level - 1))
  } else if level == 3 {
    return numbering("1.1", numbers.pos().at(level - 2), numbers.pos().at(level - 1))
  } else if level == 4 {
    return numbering(
      "1.1.1",
      numbers.pos().at(level - 3),
      numbers.pos().at(level - 2),
      numbers.pos().at(level - 1),
    )
  }
})

#show heading.where(level: 3): it => {
  counter(math.equation).update(0)
  it
}
#set math.equation(
  numbering: "1.",
  supplement: "Eq.",
)
#show math.equation.where(numbering: none): it => it.numbering("1.")
#set math.mat(delim: "[", )

#show: codly-init.with()
#codly(languages: codly-languages)
