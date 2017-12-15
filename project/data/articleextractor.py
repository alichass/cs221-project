import csv
from bs4 import BeautifulSoup
import urllib
from goose import Goose
goose = Goose()
articles = []
with open('applefoollinks.txt', 'r') as f:
    articleno = 0
    for link in f:
        articleno +=1
        if '%2Fwww.reuters.com' in link:
            link = 'https://' + link[link.find('www.reuters.com'):link.find('&ei')].replace('%2F', '/')
            print '--------------------------------------------'
            print articleno
            print link
            print '--------------------------------------------'
            r = urllib.urlopen(link).read()
            soup = BeautifulSoup(r, "lxml")
            text = soup.find('div', class_='columnLeft')
            if text is None:
                continue
            text = text.find('p')
            date = soup.find('span', class_='timestamp')
            articles.append([date.get_text().encode('utf-8'),text.get_text().encode('utf-8')])
        else:
            print '--------------------------------------------'
            print articleno
            print link
            print '--------------------------------------------'
            article = goose.extract(url=link)
            date = article.publish_date
            text = article.cleaned_text.encode('utf-8')
            if date is None or text is None:
                continue
            articles.append([date, text])

with open('applefoolarticles.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(articles)
