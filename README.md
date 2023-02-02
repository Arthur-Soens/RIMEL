# RIMEL

For the script to run : 
`apt-get install jupyter-nbconvert curl`

Retro global token : sqa_cef5900dd30278f974d984b3f3f2d5cb7a8beb4b

Retro-Bad global token : sqa_f211bc159d8bb7188d22b149ed37777ddeeabdc0

Credentials : admin / retro
Credentials bad : admin / retrobad

To analyse manually a project :
- go to Sonar and create a project and a token (I used filename)
- run `./sonar-scanner/bin/sonar-scanner -Dsonar.projectKey=token -Dsonar.sources=path_to_python.py -Dsonar.host.url=http://localhost:9000 -Dsonar.login=sqa_cef5900dd30278f974d984b3f3f2d5cb7a8beb4b`

To convert a notebook to python :
`jupyter nbconvert --to script notebook.ipynb`

You need to install nbconvert first
