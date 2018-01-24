#!/Users/plestran/Dropbox/insight/insight-env/bin/python
from flaskexample import app

app.run(ssl_context=('cert.pem', 'key.pem'))
#app.run(debug = True)
