"""Define the custom actions here"""

import sys
import ast
import json
import logging
import traceback
from io import StringIO
from typing import Any, Text, Dict, List
import pandas as pd
from pandas.core.frame import DataFrame
import requests
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from openai import OpenAI, OpenAIError
from dotenv import dotenv_values

logging.StreamHandler(stream=sys.stdout).setFormatter('%(asctime)s [%(levelname)s] %(message)s')

logger = logging.getLogger(__name__)

logger.setLevel(logging.NOTSET)
logger.propagate = False

env = dict(dotenv_values())

try:
    client = OpenAI(
        api_key=str(env.get('OPENAI_API_KEY', '')).strip(),
        max_retries=3
    )
    logger.info("OpenAI client initialized successfully")
except (OpenAIError, TypeError) as e:
    logger.info("Failed to initialize OpenAI client: %s", str(e))
    logger.info("Traceback: %s", traceback.format_exc(chain=True))
    sys.exit(1)

class RestaurantAPI:
    """Restaurant POJO"""
    def __init__(self):
        self.db: DataFrame = pd.read_csv("restaurants.csv")

    def fetch_restaurants(self) -> DataFrame:
        """Return the restaurant data."""
        # return self.db.head()
        return self.db

    def format_restaurants(self, df: DataFrame, header=True) -> Text:
        """Format the restaurant data as a CSV string."""
        # return df.to_csv(index=False, header=header)
        return df.to_json(index=False, orient='records')


class ChatGPT:
    """ChatGPT POJO"""
    def __init__(self):
        self.url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-4o-mini"
        self.headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {env.get('OPENAI_API_KEY', '').strip()}"
        }
        # self.prompt = "Answer the following question, based on the data shown. " \
        #    "Answer in a complete sentence and don't say anything else."
        self.prompt = """Answer the following question based on the restaurant data provided.

Rules:
1. Answer in complete sentences
2. If the question asks for "top N" restaurants, list them with their details
3. Format the output clearly with ratings, WiFi availability, cuisine, and distance
4. If filtering by cuisine, only show restaurants of that cuisine type
5. Sort by rating (highest first) when showing "top" or "best" restaurants
6. Be concise and helpful

Restaurant Data:
"""

    def ask(self, restaurants: DataFrame, question: str) -> str:
        """Ask a question to ChatGPT based on the restaurant data."""
        content = self.prompt + "\n\n" + restaurants + "\n\n" + question

        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}]
            )
            print(completion)
            return completion.choices[0].message.content
        except Exception as e: # pylint: disable=broad-exception-caught
            logger.info("ChatGPT ask failed: %s", str(e))
            return "Sorry, I couldn't process that request."

    def ask_old(self, restaurants: DataFrame, question: str) -> str:
        """Ask a question to ChatGPT based on the restaurant data."""
        content  = self.prompt + "\n\n" + restaurants + "\n\n" + question
        body = {
            "model":self.model, 
            "messages":[{"role": "user", "content": content}]
        }

        logger.info(self.url, self.headers, body)  # Debugging line
        result = requests.post(
            url=self.url,
            headers=self.headers,
            json=body,
            timeout=10
        )
        logger.info(json.dumps(result.json(), indent=4))  # Debugging line
        return result.json()["choices"][0]["message"]["content"]

def ask_distance(restaurant_list):
    """Ask ChatGPT to measure the distance to each restaurant."""
    if not client:
        logger.info("OpenAI client not available")
        return None

    try:
        logger.info("Calling OpenAI with restaurant list: %s...", restaurant_list[:100])
        content = "measure the least distance with each given restaurant" + '\n\n' + restaurant_list

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_measure",
                        "description": "Get the least distance",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "list of all the restaurants and distances as a dictionary(restaurant_name:distance)",
                                },
                            },
                            "required": ["location"],
                        },
                    }
                }
            ],
            tool_choice={"type": "function", "function": {"name": "get_measure"}}
        )
        logger.info("OpenAI call successful")
        return completion.choices[0].message
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.info("OpenAI call failed with error: %s", str(e))
        logger.info("Traceback: %s", traceback.format_exc(chain=True))
        return None

def get_closest_restaurant_from_csv(csv_data):
    """Get the closest restaurant directly from CSV data."""
    try:
        logger.info("Parsing CSV data: %s...", csv_data[:100])

        # df = pd.read_csv(StringIO(csv_data))
        df = pd.read_json(StringIO(csv_data))
        logger.info("DataFrame created with shape: %s", df.shape)
        logger.info("Distance column: %s", df['distance'].tolist())

        # Convert distance to numeric, handling the '05' format
        df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
        logger.info("Distance column after conversion: %s", df['distance'].tolist())

        # Find restaurant with minimum distance
        closest_idx = df['distance'].idxmin()
        closest_restaurant = df.loc[closest_idx]
        logger.info("Closest restaurant: %s with distance: %s", closest_restaurant['Restaurants'], closest_restaurant['distance'])

        return f"{closest_restaurant['Restaurants']} (Distance: {closest_restaurant['distance']} units)"
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.info("CSV parsing failed with error: %s", str(e))
        logger.info("Traceback: %s", traceback.format_exc())
        return f"Could not determine closest restaurant: {str(e)}"

try:
    restaurant_api = RestaurantAPI()
    chatGPT = ChatGPT()
    logger.info("APIs initialized successfully")
except Exception as e:
    logger.info("Warning: Could not initialize APIs: %s", e)
    logger.info("Traceback: %s", traceback.format_exc())

class ActionShowRestaurants(Action):
    """Show a list of restaurants."""
    def name(self) -> Text:
        return "action_show_restaurants"

    async def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],
            utter: bool = True) -> List[Dict[Text, Any]]:
        """
        Show a list of restaurants.
        """
        try:
            restaurants: DataFrame = restaurant_api.fetch_restaurants()
            results = restaurant_api.format_restaurants(restaurants)
            # readable = restaurant_api.format_restaurants(
            #     restaurants[['Restaurants', 'Rating']], header=False
            # )
            # dispatcher.utter_message(text=f"Here are some restaurants:\n\n{readable}")
            readable_text = ""
            for _, row in restaurants.iterrows():
                # readable_text += f"🍽️ **{row['Restaurants']}**\n"
                # readable_text += f"   ⭐ Rating: {row['Rating']}\n"
                # readable_text += f"   📡 WiFi: {'Yes' if row['Has WiFi'] else 'No'}\n"
                # readable_text += f"   🍴 Cuisine: {row['cuisine'].title()}\n"
                # readable_text += f"   📍 Distance: {row['distance']} units\n\n"
                readable_text += f"**{row['Restaurants']}**\n"
                readable_text += f"   Rating: {row['Rating']}\n"
                readable_text += f"   WiFi: {'Yes' if row['Has WiFi'] else 'No'}\n"
                readable_text += f"   Cuisine: {row['cuisine'].title()}\n"
                readable_text += f"   Distance: {row['distance']} units\n\n"

            if utter:
                dispatcher.utter_message(text=f"Here are some restaurants:\n\n{readable_text}")
            return [SlotSet("results", results)]
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.info("ActionShowRestaurants failed: %s", str(e))
            logger.info("Traceback: %s", traceback.format_exc())
            dispatcher.utter_message(text="Sorry, I couldn't fetch restaurant data.")
            return []


def get_distance(d):
    """
    Get the restaurant with the least distance.
    """
    try:
        logger.info("Processing distance data: %s, type is %s", d, type(d))
        # Handle both string and dict inputs
        if isinstance(d, str):
            d = json.loads(d)

        logger.info(d)

        for i in d.keys():
            d[i] = float(d[i])
        t = min(d, key=d.get)
        logger.info("Closest restaurant from OpenAI data: %s", t)
        return t
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.info("get_distance failed: %s", str(e))
        logger.info("Traceback: %s", traceback.format_exc())
        return "Sorry, couldn't calculate distances." + str(e)

class ActionRestaurantsDetail(Action):
    """Show details about a specific restaurant."""
    def name(self) -> Text:
        return "action_restaurants_detail"

    async def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        """
        Show details about a specific restaurant.
        """
        try:
            previous_results = tracker.get_slot("results")
            if not previous_results:
                dispatcher.utter_message(
                    text="Please show restaurants first by saying 'show me restaurants'."
                )
                return []

            # question = str(tracker.latest_message["text"])
            restaurant_entity = next(tracker.get_latest_entity_values("restaurant"), None)
            if restaurant_entity:
                # User asked about a specific restaurant
                question = f"Tell me about {restaurant_entity}"
            else:
                # Use the full user question
                question = str(tracker.latest_message["text"])

            answer = chatGPT.ask(previous_results, question)
            dispatcher.utter_message(text=answer)
            return []
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.info("ActionRestaurantsDetail failed: %s", str(e))
            logger.info("Traceback: %s", traceback.format_exc(chain=True))
            dispatcher.utter_message(text="Sorry, I couldn't get restaurant details.")
            return []


class ActionRestaurantsDistance(Action):
    """Show the distance to each restaurant."""
    def name(self) -> Text:
        return "action_distance"

    async def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        """
        Show the distance to each restaurant.
        """
        try:
            logger.info("ActionRestaurantsDistance called")
            previous_results = tracker.get_slot("results")
            if not previous_results:
                logger.info("No previous results found")
                dispatcher.utter_message(
                    text="Please show restaurants first by saying 'show me restaurants'."
                )
                return []

            logger.info("Trying OpenAI distance calculation...")
            # Try OpenAI first
            func_calling = ask_distance(previous_results)

            if func_calling and hasattr(func_calling, 'tool_calls') and func_calling.tool_calls:
                logger.info("OpenAI response received, processing...")
                # OpenAI response available
                reply_content = str(func_calling.tool_calls[0].function.arguments).strip()

                try:
                    distance_data = json.loads(reply_content)['location']
                    logger.info("Distance data extracted successfully: %s and type is %s", distance_data, type(distance_data))
                except json.JSONDecodeError as jde:
                    logger.info("JSON decoding failed: %s", str(jde))
                    # dispatcher.utter_message(text="Sorry, I couldn't parse the distance data.")
                    # return []
                    logger.info("Using ast.literal_eval for fallback....")
                    logger.info(type(ast.literal_eval(reply_content)))
                    logger.info(ast.literal_eval(reply_content))
                    distance_data = ast.literal_eval(reply_content)['location']

                result = get_distance(distance_data)
                dispatcher.utter_message(text=f"The closest restaurant is: {result}")
            else:
                logger.info("OpenAI failed, falling back to CSV parsing...")
                # Fallback to direct CSV parsing
                result = get_closest_restaurant_from_csv(previous_results)
                dispatcher.utter_message(text=f"The closest restaurant is: {result}")

            return []
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.info("ActionRestaurantsDistance failed: %s", str(e))
            logger.info("Traceback: %s", traceback.format_exc(chain=True))
            # If everything fails, try one more simple fallback
            try:
                logger.info("Attempting final fallback...")
                previous_results = tracker.get_slot("results")
                result = get_closest_restaurant_from_csv(previous_results)
                dispatcher.utter_message(text=f"The closest restaurant is: {result}")
            except Exception as e2:  # pylint: disable=broad-exception-caught
                logger.info("Final fallback also failed: %s", str(e2))
                logger.info("Traceback: %s", traceback.format_exc(chain=True))
                dispatcher.utter_message(text=f"Sorry, I couldn't calculate distances: {str(e)}")
            return []
