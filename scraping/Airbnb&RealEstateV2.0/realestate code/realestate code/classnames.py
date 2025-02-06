import requests
from bs4 import BeautifulSoup

# Configure headers and target URL
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Accept-Language': 'en-US,en;q=0.9',
}

URL = "https://www.realestate.com.lb/en/buy-apartment-house-lebanon?pg=1&sort=featured&ct=1"


def extract_class_names():
    try:
        # Fetch the webpage
        response = requests.get(URL, headers=headers)
        response.raise_for_status()

        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all elements with class attribute
        elements_with_classes = soup.find_all(class_=True)

        # Collect unique class names
        class_names = set()
        for element in elements_with_classes:
            classes = element.get('class')
            if classes:
                class_names.update(classes)

        # Save to a text file
        with open('class_names.txt', 'w') as f:
            f.write("All Class Names Found:\n\n")
            for cls in sorted(class_names):
                f.write(f"{cls}\n")

        print(f"Found {len(class_names)} unique class names. Saved to class_names.txt")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    extract_class_names()