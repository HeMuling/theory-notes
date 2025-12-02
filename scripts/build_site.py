import os
import re
import subprocess
import shutil
import sys
from datetime import datetime

# Configuration
DIRS = ['math', 'stats', 'ml', 'neuro']
BUILD_DIR = 'build'
TEMPLATE_FILE = 'index.html'
# Use locally converted TTFs to avoid relying on system fonts
FONT_PATH = os.path.join('assets', 'fonts')

# Capture both include and import to reconstruct declared order
INCLUDE_PATTERN = re.compile(r'#(?:include|import)\s+"([^"]+\.typ)"')

def get_title(filepath):
    # Default to filename capitalized (e.g. "bayesian-optimization" -> "Bayesian Optimization")
    filename = os.path.basename(filepath)
    name_no_ext = os.path.splitext(filename)[0]
    title = name_no_ext.replace('-', ' ').title()
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # Look for #hd1("Title") or #hd2("Title")
            # Adjust regex to match standard typst function calls
            match = re.search(r'#hd[1-2]\("([^"]+)"\)', content)
            if match:
                title = match.group(1)
    except Exception as e:
        print(f"Warning: Could not read title from {filepath}: {e}")
        
    return title

def get_include_order(index_path):
    order = []
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            content = f.read()
            for match in INCLUDE_PATTERN.finditer(content):
                inc = os.path.basename(match.group(1))
                order.append(inc)
    except Exception as e:
        print(f"Warning: Could not parse include order from {index_path}: {e}")
    return order

def compile_typst(input_path, output_rel_path):
    output_path = os.path.join(BUILD_DIR, output_rel_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Compiling {input_path} -> {output_path}")
    try:
        # Use --root . to ensure imports work correctly from project root
        # Pass root-file so sections can optionally decide whether to render local bibliographies
        subprocess.run([
            "typst",
            "compile",
            "--root", ".",
            "--font-path", FONT_PATH,
            "--ignore-system-fonts",
            "--input", f"root-file={input_path}",
            input_path,
            output_path,
        ], check=True)
    except subprocess.CalledProcessError:
        print(f"Error compiling {input_path}")
        # Don't exit, try to continue with other files

def main():
    # Clean/Create build dir
    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
    os.makedirs(BUILD_DIR)

    nav_html = []
    build_timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    
    # 1. Overview (main.typ)
    if os.path.exists("main.typ"):
        compile_typst("main.typ", "main.pdf")
        nav_html.append('<div class="nav-group-title">Overview</div>')
        # First item active by default
        nav_html.append(f'''<a href="main.pdf" target="preview-frame" class="nav-item active" onclick="setActive(this)">Complete Notes</a>''')

    # 2. Subjects
    for d in DIRS:
        if not os.path.exists(d):
            continue
            
        # Section Header
        section_title = d.upper() if d == 'ml' else d.capitalize()
        if d == 'neuro': section_title = 'Neuroscience'
        if d == 'math': section_title = 'Mathematics'
        if d == 'stats': section_title = 'Statistics'
        if d == 'ml': section_title = 'Machine Learning'
        
        nav_html.append(f'<div class="nav-group-title">{section_title}</div>')
        
        # Index file
        index_typ = os.path.join(d, "index.typ")
        include_order = []
        if os.path.exists(index_typ):
            include_order = get_include_order(index_typ)
            overview_pdf = f"{d}/overview.pdf"
            compile_typst(index_typ, overview_pdf)
            nav_html.append(f'''<a href="{overview_pdf}" target="preview-frame" class="nav-item" onclick="setActive(this)">{section_title} Overview</a>''')

        # Chapter files: Scan for *.typ in the directory (non-recursive)
        files = []
        for f in os.listdir(d):
            if f.endswith(".typ") and f != "index.typ":
                files.append(f)
        
        # Preserve the order declared in index.typ; any extras fall back to alphabetical
        files.sort(key=lambda name: (name not in include_order, include_order.index(name) if name in include_order else name))
        
        for f in files:
            filepath = os.path.join(d, f)
            title = get_title(filepath)
            pdf_rel_path = f"{d}/{os.path.splitext(f)[0]}.pdf"
            
            compile_typst(filepath, pdf_rel_path)
            
            # Add to nav
            # Use nav-sub-item style to differentiate from the main index
            nav_html.append(f'''<a href="{pdf_rel_path}" target="preview-frame" class="nav-item nav-sub-item" onclick="setActive(this)">{title}</a>''')

    # 3. Generate HTML
    if os.path.exists(TEMPLATE_FILE):
        with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f:
            template = f.read()
        
        final_html = template.replace('<!-- GENERATED_NAV_LINKS -->', '\n'.join(nav_html))
        final_html = final_html.replace('<!-- GENERATED_LAST_BUILT -->', build_timestamp)
        
        with open(os.path.join(BUILD_DIR, 'index.html'), 'w', encoding='utf-8') as f:
            f.write(final_html)
            
        # Copy assets if any (e.g. if CSS was external, but it's embedded now)
        # If there are images referenced in HTML, they need to be copied? 
        # The current HTML has no external assets. 
        # But PDF might reference assets. Typst handles embedding images in PDF.
        # So we just need the PDFs and the index.html in build/.
    else:
        print(f"Error: {TEMPLATE_FILE} not found.")
        sys.exit(1)

    print("Build complete.")

if __name__ == "__main__":
    main()
