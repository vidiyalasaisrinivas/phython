from bs4 import BeautifulSoup
import requests

def searchInURL():
    #assigning the Required web page to the url variable
    source= "https://en.wikipedia.org/wiki/Deep_learning"
    #the data in the URL is converted into the text and save in the response
    response= requests.get(source).text
    #parsing the data into beautiful soup in lxml format
    soup= BeautifulSoup(response, "html.parser")

    # pretty-printing(to print the data in the HTML format)
    # soup.prettify()

    # Printing the title
   # wikititle =soup.title
    wikititle = soup.title.text
    print("Title is:",wikititle)

    # Writing results into a file
    with open("output1.html", "a") as file:
        file.write(str(wikititle))


    #looping through the all a tags
    for links in soup.find_all('a'):
        # fetching href's
        hrefs=links.get('href')
        with open("output1.html", "a") as file:
            file.write(str(hrefs))
            file.write("\n")
        print(hrefs)



if __name__== '__main__':
    #function call
    searchInURL()
