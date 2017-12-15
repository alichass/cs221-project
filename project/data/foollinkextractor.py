from bs4 import BeautifulSoup
import urllib
import time
from selenium import webdriver

def pulllinks(mainpage, filepath):
    driver = webdriver.Chrome('C:\Users\Ali Chassebi\Documents\Computery Shit\chromedriver')
    driver.get(mainpage)
    count = 0
    print "Looping through https://www.fool.com/quote/nasdaq/alphabet-c-shares/goog/content"
    while count <25:
        button_element = driver.find_element_by_css_selector('body > div.main-container > div.page-grid-container > div > section > section > div.section-more-link-container > a')
        if button_element == None:
            break
        count +=1
        button_element.click()
        time.sleep(2)
        print count
    print "Looped back far enough"
    print "-------------------------"
    print "Finding divs"
    linkdivs = driver.find_elements_by_class_name("article-link")
    # # soup = BeautifulSoup(driver.page_source)
    # # linkdivs = soup.find_all('a', class_='article-link')
    print "Divs Found"
    print "-------------------------"
    links = []
    print "Finding " + str(len(linkdivs)) +" links"
    count = 0
    for div in linkdivs:
        count+=1
        print count
        links.append(div.get_attribute("href"))

    print "Links Found"
    print "-------------------------"
    print "Writing " + str(len(links)) + " links to csv"

    with open(filepath, 'w') as f:
      for link in links:
          f.write(link + '\n')

mainpage = 'https://www.fool.com/quote/nasdaq/alphabet-c-shares/goog/content'
filepath = 'googlefoollinks.txt'
pulllinks(mainpage, filepath)
