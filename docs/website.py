#!/usr/bin/env python3
import os

# This script scans the current directory for report files and generates an
# index.html with grouped links by dataset name.

def main():
    script_name = os.path.basename(__file__)
    exclude = {script_name, 'index.html'}

    # All relevant files
    files = [f for f in os.listdir('.')
             if f not in exclude and os.path.isfile(f)
             and os.path.splitext(f)[1].lower() in ('.png', '.html')]

    # Map file suffixes to human-readable labels
    suffix_map = {
        '_utility_spmf_search_only.png': 'Search Only',
        '_search_only.png': 'Search Only',
        '_utility_spmf_speedup_plot.html': 'GPU Dynamic Parallelism Sweep Speedup Plot',
        '_speedup_plot.html': 'GPU Dynamic Parallelism Sweep Speedup Plot',
        '_utility_spmf_search_time_plot.html': 'Search Time Plot',
        '_search_time_plot.html': 'Search Time Plot',
        '_utility_spmf_clustered_by_util.png': 'Execution time',
        '_clustered_by_util.png': 'Execution time',
    }
    # Desired link order
    link_order = ['Search Only', 'GPU Dynamic Parallelism Sweep Speedup Plot', 'Search Time Plot', 'Execution time']

    # Group files by dataset prefix
    groups = {}
    for fname in files:
        for suffix, label in suffix_map.items():
            if fname.endswith(suffix):
                dataset = fname[:-len(suffix)]
                groups.setdefault(dataset, {})[label] = fname
                break

    # Start building HTML
    html_lines = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head>',
        '    <meta charset="UTF-8">',
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        '    <title>Index of Reports</title>',
        '    <style>',
        '        body { font-family: sans-serif; padding: 20px; }',
        '        h2 { color: black; margin-top: 1.5em; }',
        '        ul { list-style-type: none; padding-left: 1em; }',
        '        li { margin: 0.5em 0; }',
        '        a { text-decoration: none; color: #0366d6; }',
        '        a:hover { text-decoration: underline; }',
        '    </style>',
        '</head>',
        '<body>',
        '    <h1>Index of Reports</h1>',
    ]

    # Add grouped sections
    for dataset in sorted(groups):
        html_lines.append(f'    <h2>{dataset}</h2>')
        html_lines.append('    <ul>')
        for label in link_order:
            if label in groups[dataset]:
                fname = groups[dataset][label]
                html_lines.append(f'        <li><a href="{fname}">{label}</a></li>')
        html_lines.append('    </ul>')

    # Close HTML
    html_lines += [
        '</body>',
        '</html>'
    ]

    # Write to index.html
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_lines))

    print('index.html has been generated successfully.')

if __name__ == '__main__':
    main()
