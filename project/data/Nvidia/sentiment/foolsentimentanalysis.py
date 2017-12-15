import csv
import requests
from  dateutil.parser import parse

date_sent = []
with open('nvidiafoolarticles.csv', 'r') as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
        i += 1
        print '----------------------------------------------------------------'
        print i
        r = requests.post("http://text-processing.com/api/sentiment/", data=[('text', row[1]),])
        if r is None:
            continue
        print r.text
        print '---------------------------------------------------------------------- '
        dt = parse(str(row[0]))
        date_sent.append([dt.strftime('%Y-%m-%d'), r.json()['probability']['neutral'], r.json()['probability']['neg'], r.json()['probability']['pos']])

with open('nvidiafoolsentiment.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(date_sent)
