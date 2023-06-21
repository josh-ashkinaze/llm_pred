"""
Author: Joshua Ashkinaze
Date: 2023-06-20

Description: This script gets binary political events from the Metaculus API and saves it to a csv file.
"""

import pandas as pd
import requests
import os
from bs4 import BeautifulSoup
import json
import numpy as np
import concurrent.futures
import datetime
import argparse
import logging


def unix_timestamp_to_timestamp(unix_timestamp):
    timestamp = datetime.datetime.fromtimestamp(unix_timestamp)
    timestamp_str = timestamp.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    return timestamp_str


def get_predictions(ts, min_threshold):
    """
    Get predictions from metaculus.

    Note: We want to get `initial' forecasts from humans but the
    very first forecast will have like 1 person. So I get the earliest forecast that has
    `min_threshold` people where min_threshold can be an int or a float that is a proportion of the final number of votes
    (i.e: If min_threshold=0.2, then it's the earliest forecast that had at least 20% x final number of votes).

    """
    try:
        final_pred = ts[-1]
        final_pred_time = unix_timestamp_to_timestamp(final_pred['t'])
        final_pred_pred = final_pred['community_prediction']
        final_pred_n = final_pred['num_predictions']

        if isinstance(min_threshold, float):
            min_threshold = int(min_threshold * final_pred_n)
        else:
            pass

        # Now find fist week where there's some minimum of votes
        init_pred = None
        for week in ts:
            if week['num_predictions'] >= min_threshold:
                init_pred = week
                break

        if init_pred:
            init_pred_time = unix_timestamp_to_timestamp(init_pred['t'])
            init_pred_pred = init_pred['community_prediction']
            init_pred_n = init_pred['num_predictions']
        else:
            init_pred_time = np.NaN
            init_pred_pred = np.NaN
            init_pred_n = np.NaN

        return {
            'final_pred_time': final_pred_time,
            'final_pred_conf': final_pred_pred,
            'final_pred_n': final_pred_n,
            'init_pred_time': init_pred_time,
            'init_pred_conf': init_pred_pred,
            'init_pred_n': init_pred_n
        }
    except Exception as e:
        print(f"I got an error {e} when parsing ts {ts}")
        return {'final_pred_time': np.NaN,
                'final_pred_conf': np.NaN,
                'final_pred_n': np.NaN,
                'init_pred_time': np.NaN,
                'init_pred_conf': np.NaN,
                'init_pred_n': np.NaN}


def parse_more_fields(url):
    try:
        response = requests.get(url)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        data = json.loads(soup.text)
        result = {}
        for key in ['description', 'resolution_criteria', 'fine_print']:
            if key in data and data[key]:
                result[key] = data[key]
            else:
                result[key] = np.nan
    except Exception as e:
        print(e)
        result = {'description': np.nan, 'resolution_criteria': np.nan, 'fine_print': np.nan}
    return result


def parse_event(f, min_threshold=50):
    data = {}
    cats = f['categories']
    data['url'] = "https://www.metaculus.com" + f['page_url']
    data['id'] = f['id']
    data['title'] = f['title']
    data['categories'] = [{k: cat[k] for k in ['id', 'short_name', 'long_name']} for cat in cats]

    data['publish_time'] = f['publish_time']
    data['created_time'] = f['created_time']
    data['closed_time'] = f['close_time']
    data['resolve_time'] = f['resolve_time']
    data['votes'] = f['votes']
    data['n_forecasters'] = f['number_of_forecasters']
    data['prediction_count'] = f['prediction_count']
    data['resolution'] = f['resolution']
    data['prediction_timeseries'] = f['prediction_timeseries']
    data.update(get_predictions(f['prediction_timeseries'], min_threshold))
    return data


def fetch_and_parse_event(event):
    parsed_event = parse_event(event)
    more_fields = parse_more_fields(event['url'])
    parsed_event.update(more_fields)
    return parsed_event


def main(min_threshold, resolve_time_lt, publish_time_gt):
    LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(filename=f'{os.path.basename(__file__)}.log', level=logging.INFO, format=LOG_FORMAT,
                        datefmt='%Y-%m-%d %H:%M:%S', filemode='w')

    logging.info(
        f"Starting pull with min_threshold {min_threshold}, resolve_time_lt {resolve_time_lt}, publish_time_gt {publish_time_gt}")
    with open('../secrets/secrets.json', 'r') as f:
        secrets = json.load(f)
    key = secrets['meta_key']

    parsed_events = []

    url = "https://www.metaculus.com/api2/questions/"
    parameters = {
        "resolve_time__lt": resolve_time_lt,
        "publish_time__gt": publish_time_gt,
        "type": "forecast",
        "forecast_type": "binary",
        "resolved": True,
        "limit": 100,
        "include_description": "true",
    }

    headers = {
        "Authorization": "Token {}".format(key),
    }

    num_events_to_fetch = 6000
    num_events_fetched = 0
    next_url = url

    with concurrent.futures.ThreadPoolExecutor() as executor:
        while next_url and num_events_fetched < num_events_to_fetch:
            response = requests.get(next_url, params=parameters, headers=headers)
            results = response.json()
            events = results["results"]

            for parsed_event in executor.map(fetch_and_parse_event, events):
                parsed_events.append(parsed_event)
                num_events_fetched += 1

                if num_events_fetched >= num_events_to_fetch:
                    break

            next_url = results["next"]

    logging.info(f"Total events fetched: {num_events_fetched}")

    events = pd.DataFrame(parsed_events)
    events.to_csv(f"../data/metaculus_events.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Get binary political events from the Metaculus API and save to a CSV file.")
    parser.add_argument("--min_threshold", type=int, default=50, help="Minimum threshold for initial predictions.")
    parser.add_argument("--to_date", type=str, default="2023-03-01", help="Resolve time upper limit.")
    parser.add_argument("--from_date", type=str, default="2021-10-01", help="Publish time lower limit.")
    args = parser.parse_args()
    main(args.min_threshold, args.to_date, args.from_date)
