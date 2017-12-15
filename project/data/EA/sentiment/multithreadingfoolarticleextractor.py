import csv
from bs4 import BeautifulSoup
import urllib
from multiprocessing.dummy import Pool  # This is a thread-based Pool
from multiprocessing import cpu_count
import time

articles = []
def crawltoCSV(urlrec):
        print urlrec
        r = urllib.urlopen(urlrec)
        soup = BeautifulSoup(r, "lxml")
        r.close()
        articlecont = soup.find('span', class_='article-content')
        if articlecont is None:
            return []

        allparagraphs = articlecont.find_all('p')
        alltext = ''
        for paragraph in allparagraphs:
            alltext += paragraph.get_text() + '\n'
        date = soup.find('div', 'publication-date')
        return [date.get_text().encode('utf-8'), alltext.encode('utf-8')]

pool = Pool(cpu_count() * 4)
with open('eafoollinks.txt', 'r') as f:
    results = pool.map(crawltoCSV, f)
with open('eafoolarticles.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(results)
