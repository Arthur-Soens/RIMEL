import os
import subprocess
import requests


def download_repositories(number: int, page_number: int) -> None:
    token = "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"
    url = "https://api.github.com/search/repositories?q=language:jupyter-notebook&order=desc&page=" + str(page_number)
    headers = {
        'Accept': 'application/vnd.github.preview.text-match+json',
        'Authorization': 'Bearer ' + token,
    }

    resp = requests.get(url, headers=headers)

    # Get urls of one of each three repository
    urls = []
    for i in range(0, number * 3, 3):
        urls.append(resp.json()['items'][i]['html_url'])

    print(urls)

    for url in urls:
        subprocess.run(['git', 'clone', '--depth', '1', url, './repositories/' + url.split('/')[-1]])


def download_good_repositories(number: int) -> None:
    print(f'Downloading {number} repositories from GitHub...')
    download_repositories(number, 1)
    print('Good repositories downloaded !')


def download_bad_repositories(number: int) -> None:
    print(f'Downloading {number} repositories from GitHub...')
    download_repositories(number, 30)
    print('Bad repositories downloaded !')


def extract_jupyter_notebooks() -> None:
    print("Extracting jupyter notebooks from repositories...")
    subprocess.run(['find', './repositories/', '-name', '*.ipynb', '-exec', 'cp', '{}', './jupyter-notebooks/', ';'])
    print("Jupyter notebooks extracted !")


def convert_notebooks_to_python() -> None:
    print("Converting jupyter notebooks to scripts...")
    try:
        subprocess.run(['jupyter', 'nbconvert',
                        './jupyter-notebooks/*.ipynb',
                        '--to', 'script',
                        '--output-dir', './python-scripts/'])
    except (Exception, RuntimeError):
        print("Notebook validation failed !")

    print("Jupyter notebooks converted to scripts !")

    print("Removing non python files...")
    for item in os.listdir('./python-scripts/'):
        if not item.endswith(".py"):
            os.remove(os.path.join('./python-scripts/', item))
    print("Non python files removed !")


def scan_python_files() -> None:
    print("Scanning python files for code quality...")
    for file in os.listdir('./python-scripts/'):
        subprocess.run(['curl', '-X', 'POST', 'http://localhost:9000/api/projects/create?name=' + file[:-3]
                        + '&project=' + file[:-3]])

    for file in os.listdir('./python-scripts/'):
        subprocess.run(['./sonar-scanner/bin/sonar-scanner',
                        '-Dsonar.projectKey=' + file[:-3],
                        '-Dsonar.sources=./python-scripts/' + file,
                        '-Dsonar.host.url=http://localhost:9000',
                        '-Dsonar.login=sqa_cef5900dd30278f974d984b3f3f2d5cb7a8beb4b'])
    print("Python files scanned !")


def scan_bad_python_files() -> None:
    print("Scanning python files for code quality...")
    for file in os.listdir('./python-scripts/'):
        subprocess.run(['curl', '-X', 'POST', 'http://localhost:9000/api/projects/create?name=' + file[:-3]
                        + '&project=' + file[:-3]])

    for file in os.listdir('./python-scripts/'):
        subprocess.run(['./sonar-scanner/bin/sonar-scanner',
                        '-Dsonar.projectKey=' + file[:-3],
                        '-Dsonar.sources=./python-scripts/' + file,
                        '-Dsonar.host.url=http://localhost:9001',
                        '-Dsonar.login=sqa_f211bc159d8bb7188d22b149ed37777ddeeabdc0'])
    print("Python files scanned !")


def delete_files() -> None:
    print("Deleting files from folders { ./jupyter-notebooks/, ./python-scripts/ }")

    for file in os.listdir('./python-scripts/'):
        os.remove('./python-scripts/' + file)

    for file in os.listdir('./jupyter-notebooks/'):
        os.remove('./jupyter-notebooks/' + file)

    print("All files deleted !")


if __name__ == '__main__':
    download_good_repositories(1)
    # download_bad_repositories(10)
    extract_jupyter_notebooks()
    convert_notebooks_to_python()
    scan_python_files()
    # scan_bad_python_files()
    # delete_files()
