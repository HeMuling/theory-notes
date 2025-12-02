# Theory Notes

个人使用的理论笔记，使用 [Typst](https://github.com/typst/typst) 编写与导出。

## 本地构建
1. 确保已安装 `typst`（0.14+）与 `python3`。
2. 单文件编译：`typst compile main.typ --font-path assets/fonts`。
3. 运行 `python3 scripts/build_site.py`，生成的 HTML 与 PDF 位于 `build/`。
   - 章节顺序跟随各目录下的 `index.typ` 中的 `#include` 顺序。
   - 大章节汇总 PDF 命名为 `overview.pdf`（例如 `build/ml/overview.pdf`）。

## 本地模拟 GitHub Action
运行 `bash scripts/test_github_action.sh`，流程与工作流 `typst-pages.yml` 对齐。

## 结构提示
- `prelude.typ`：全局样式与包导入。
- `main.typ`：总入口，包含大纲、各章节、参考文献与索引。
- `scripts/build_site.py`：批量编译并生成 `build/index.html` 导航。
