from langflow.custom import Component
from langflow.io import (
    StrInput,
    IntInput,
    SecretStrInput,
    Output,
    FloatInput,
    BoolInput,
)
from langflow.schema import DataFrame
import requests
import pandas as pd
import time
import re


class GooglePlacesSearch(Component):
    display_name = "Google Places Search"
    description = "Search Google Places API with filters and optional email scraping, validation, and limits."
    icon = "mdi-map-search"
    name = "GooglePlacesSearch"
    field_order = [
        "query",
        "max_results",
        "min_rating",
        "max_price_level",
        "scrape_emails",
        "max_emails",
        "api_key",
    ]

    inputs = [
        StrInput(
            name="query",
            display_name="Search Query",
            info="Google Places search query (e.g., 'catering services near Paris').",
            required=True,
        ),
        IntInput(
            name="max_results",
            display_name="Max Results",
            info="Maximum number of results to retrieve (up to 60).",
            value=20,
        ),
        FloatInput(
            name="min_rating",
            display_name="Minimum Rating",
            info="Minimum rating to include (e.g., 4.0).",
            value=0.0,
        ),
        IntInput(
            name="max_price_level",
            display_name="Max Price Level",
            info="Maximum acceptable price level (0=free, 4=very expensive).",
            value=4,
        ),
        BoolInput(
            name="scrape_emails",
            display_name="Scrape Emails from Website",
            info="Enable scraping of business websites for email addresses.",
            value=True,
        ),
        IntInput(
            name="max_emails",
            display_name="Max Emails per Site",
            info="Maximum number of emails to extract per website.",
            value=3,
        ),
        SecretStrInput(
            name="api_key",
            display_name="Google API Key",
            info="Your Google Places API key.",
            required=True,
        ),
    ]

    outputs = [
        Output(name="results", display_name="Search Results", method="search_results")
    ]

    def get_place_details(self, place_id: str) -> dict:
        url = "https://maps.googleapis.com/maps/api/place/details/json"
        fields = "formatted_phone_number,international_phone_number,website,opening_hours"
        params = {
            "place_id": place_id,
            "fields": fields,
            "key": self.api_key,
        }
        try:
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            return resp.json().get("result", {})
        except Exception as e:
            self.log(f"Error fetching details for {place_id}: {repr(e)}")
            return {}

    def extract_valid_emails_from_website(self, url: str) -> str | None:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code != 200 or not response.text:
                return None

            raw_emails = re.findall(
                r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", response.text
            )
            valid_emails = [
                email for email in set(raw_emails)
                if not any(
                    invalid in email.lower()
                    for invalid in ["example", ".png", ".jpg", ".jpeg", ".svg", ".css", ".js"]
                )
            ]
            valid_emails = sorted(valid_emails)[: self.max_emails]
            return ", ".join(valid_emails) if valid_emails else None
        except Exception as e:
            self.log(f"Failed to scrape {url}: {repr(e)}")
            return None

    def search_results(self) -> DataFrame:
        try:
            base_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
            params = {"query": self.query, "key": self.api_key}

            all_results = []
            fetched = 0
            max_fetch = min(self.max_results or 20, 60)

            while fetched < max_fetch:
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()

                results = data.get("results", [])
                if not results:
                    break

                for place in results:
                    rating = place.get("rating", 0)
                    price_level = place.get("price_level", 0)

                    if rating < self.min_rating or price_level > self.max_price_level:
                        continue

                    place_id = place.get("place_id")
                    details = self.get_place_details(place_id)
                    website = details.get("website")
                    email = None

                    if self.scrape_emails and website:
                        email = self.extract_valid_emails_from_website(website)

                    all_results.append(
                        {
                            "name": place.get("name"),
                            "address": place.get("formatted_address"),
                            "vicinity": place.get("vicinity"),
                            "rating": rating,
                            "user_ratings_total": place.get("user_ratings_total"),
                            "price_level": price_level,
                            "business_status": place.get("business_status"),
                            "open_now": place.get("opening_hours", {}).get("open_now"),
                            "opening_hours_weekly": "; ".join(
                                details.get("opening_hours", {}).get("weekday_text", [])
                            ),
                            "types": ", ".join(place.get("types", [])),
                            "latitude": place.get("geometry", {})
                            .get("location", {})
                            .get("lat"),
                            "longitude": place.get("geometry", {})
                            .get("location", {})
                            .get("lng"),
                            "photo_reference": place.get("photos", [{}])[0].get(
                                "photo_reference"
                            ),
                            "place_id": place_id,
                            "icon": place.get("icon"),
                            "plus_code": place.get("plus_code", {}).get("global_code"),
                            "phone_number": details.get("formatted_phone_number")
                            or details.get("international_phone_number"),
                            "website": website,
                            "emails": email,
                        }
                    )
                    fetched += 1
                    if fetched >= max_fetch:
                        break

                next_page_token = data.get("next_page_token")
                if not next_page_token or fetched >= max_fetch:
                    break

                time.sleep(2)
                params = {"pagetoken": next_page_token, "key": self.api_key}

            if not all_results:
                self.status = "No results found."
                return DataFrame(pd.DataFrame({"message": ["No results found."]}))

            return DataFrame(pd.DataFrame(all_results))

        except Exception as e:
            self.status = f"Error: {str(e)}"
            self.log(repr(e))
            return DataFrame(pd.DataFrame({"error": [str(e)]}))
