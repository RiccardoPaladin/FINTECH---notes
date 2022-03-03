"""
Exercise on get and post requests from wesite
"""


import requests

recr  = requests.get("http://www.spacejam.com/1996")
recr.status_code      #it has to show 200
recr.content
recr.cookies


rec = requests.get("https://finance.yahoo.com/quote/AAPL/?fr=sycsrp_catchall")
rec.status_code
if rec.status_code == 200:
    print('success')
else:
    print('error:' %f/ rec.status_code)

rec.content
rec.url
rec.text
rec.headers




requests.post()     #post is another kind of request
