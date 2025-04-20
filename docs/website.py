import os

def main():
    # Exclude this script and any existing index.html
    script_name = os.path.basename(__file__)
    exclude = {script_name, 'index.html'}

    # Collect relevant files
    files = [f for f in os.listdir('.')
             if f not in exclude and os.path.splitext(f)[1].lower() in ('.png', '.html')]
    files.sort()

    # Build HTML content
    html_lines = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head>',
        '    <meta charset="UTF-8">',
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        '    <title>Index of Reports</title>',
        '</head>',
        '<body>',
        '    <h1>Index of Reports</h1>',
        '    <ul>'
    ]

    for fname in files:
        # Link display name is the file name
        html_lines.append(f'        <li><a href="{fname}">{fname}</a></li>')

    html_lines += [
        '    </ul>',
        '</body>',
        '</html>'
    ]

    # Write to index.html
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_lines))

    print('index.html has been generated successfully.')

if __name__ == '__main__':
    main()
