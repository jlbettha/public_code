import json

import requests


def fetch_data(*, update: bool = False, json_cache: str, url: str) -> dict:
    """
    Betthauser, 2024 - This type of local caching is best practice.

    *** If update = False:
        New data is acquired via API request (usually costs $).
        BUT if you already have the data cached, then it
        will simply fetch it from local storage.

    *** if update = True:
        Forces acquisition of data from the API. Use if data may have been updated since prior caching.

    Returns JSON data "json_cache" from local or "url"

    """
    if update:
        json_data = None
    else:
        try:
            with open(json_cache) as file:
                json_data = json.load(file)
                print("Fetched data from local cache.")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"No local cache found: {e}")
            json_data = None

    if not json_data:
        print("Fetching new json data... (creating local cache)")
        json_data = requests.get(url).json()
        with open(json_cache, "w") as file:
            json.dump(json_data, file, indent=4)

    return json_data


def main() -> None:
    api_url: str = "https://dummyjson.com/comments"
    cache_file: str = "json_cache/comments.json"
    _: dict = fetch_data(update=False, json_cache=cache_file, url=api_url)


if __name__ == "__main__":
    main()
