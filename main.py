import subprocess
import requests


def download_repositories(number: int) -> None:
    print(f'Downloading {number} repositories from GitHub...')
    token = "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"
    url = "https://api.github.com/search/repositories?q=language:jupyter-notebook&order=desc"
    headers = {
        'Accept': 'application/vnd.github.preview.text-match+json',
        'Authorization': 'Bearer ' + token,
    }

    resp = requests.get(url, headers=headers)
    print(resp.status_code)

    urls = [item['html_url'] for item in resp.json()['items'][:number]]
    print(urls)

    for url in urls:
        subprocess.run(['git', 'clone', '--depth', '1', url, './repositories/' + url.split('/')[-1]])

    print(f'Repositories downloaded !')


def extract_jupyter_notebooks() -> None:
    print("Extracting jupyter notebooks from repositories...")
    subprocess.run(['mv', './repositories/**/*.ipynb', './jupyter-notebooks/'])
    print("Jupyter notebooks extracted !")


def convert_notebooks_to_python():
    print("Converting jupyter notebooks to python scripts...")
    subprocess.run(['jupyter', 'nbconvert',
                    './jupyter-notebooks/*.ipynb',
                    '--to', 'script',
                    '--output-dir', './python-scripts/'])

    print("Jupyter notebooks converted to python scripts !")


def scan_python_files():
    print("Scanning python files for code quality...")
    subprocess.run(['./sonar-scanner/bin/sonar-scanner',
                    '-Dsonar.projectKey=rimel',
                    '-Dsonar.sources=./python-scripts',
                    '-Dsonar.host.url=http://127.0.0.1:9000',
                    '-Dsonar.login=sqp_eca496bcb5a1a655ee3f56ef498e489c402012e7'])
    print("Python files scanned !")


def delete_files() -> None:
    print("Deleting files from folders { ./repositories/, ./jupyter-notebooks/, ./python-scripts/ }")
    subprocess.run(['rm', '-rf', './repositories/'])
    subprocess.run(['rm', '-rf', './jupyter-notebooks/'])
    subprocess.run(['rm', '-rf', './python-scripts/'])
    print("All files deleted !")


if __name__ == '__main__':
    download_repositories(15)
    extract_jupyter_notebooks()
    convert_notebooks_to_python()
    scan_python_files()
    delete_files()

