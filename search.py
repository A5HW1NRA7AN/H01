from duckduckgo_search import DDGS

class Search():
    def __init__(self, query, max_results=5):
        self.query = query
        self.max_results = max_results
        self.url = []
        self.title = []

    def search(self):
        results = DDGS().text(self.query, max_results=self.max_results)
        for data in results:
            self.url.append(data['href'])
            self.title.append(data['title'])
        return dict(zip(self.title, self.url))

if __name__ == "__main__":
    search_obj = Search("SDOH data", 100)
    results = search_obj.search()
    for title, url in results.items():
        print(f"Title: {title}, URL: {url}\n")