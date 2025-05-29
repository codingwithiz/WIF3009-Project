import requests
import json;
import sys
import io
import csv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

headers = {
    'x-rapidapi-key': "d19af7e82dmshd52437477f45e28p13ab87jsn42a7b0482a74",
    'x-rapidapi-host': "twitter-api45.p.rapidapi.com"
}

usernames = [
    "Dream",
    "Ninja",
    "pewdiepie",
    "Valkyrae",
    "shroud",
    "pokimanelol",
    "DrDisrespect",
    "Jacksepticeye",
    "thaRadBrad",
    "bugha",
    "ishowspeedsui",
    "TheGrefg",
    "MrBeast"
    ]


def get_user_info(usernames, headers):
    
    user_data_list = []
    url = "https://twitter-api45.p.rapidapi.com/screenname.php"

    for username in usernames:
        try:
            querystring = {"screenname": username}
            response = requests.get(url, headers=headers, params=querystring)
            data = response.json()
            user_info = {
                "name": data.get("name"),
                "username": data.get("profile"),
                "profile_status": data.get("status"),
                "user_id": data.get("rest_id"),
                "verified": data.get("blue_verified"),
                "followers": data.get("sub_count"),
                "friends": data.get("friends"),
                "statuses": data.get("statuses_count"),
                "media_count": data.get("media_count"),
                # "retweet_average:" :retweet_average , 
                # "likes_average:" :like_average , 
                # "replies_average:" :replies_average , 

            }

            print(json.dumps(user_info, indent=4))  # Optional: can be removed
            user_data_list.append(user_info)

        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON for user {username}: {e}")
        except Exception as e:
            print(f"Error processing user {username}: {e}")

    return user_data_list

def get_user_tweets(usernames, headers):
    url = "https://twitter-api45.p.rapidapi.com/usermedia.php"
    user_tweets_list = []

    for username in usernames:
        try:

            retweet_total = 0
            like_total = 0
            tweet_count = 0
            replies_total = 0
            
            querystring = {"screenname": username}
            response = requests.get(url, headers=headers, params=querystring)
            response.encoding = 'utf-8'
            data = response.json()
            media_entries = []

            # Check if the response has media data
            if "timeline" in data:
                    # Get media URLs if present
                for tweet in data["timeline"]:
                    tweet_data = {
                        "text": tweet["text"],
                        "views": tweet["views"],
                        "date_created":tweet["created_at"],
                        "replies": tweet["replies"],
                        "retweets": tweet["retweets"],
                        "likes": tweet["favorites"]
                    }

                    retweet_total += int(tweet["retweets"])
                    like_total += int(tweet["favorites"])
                    replies_total += int(tweet["replies"])
                    tweet_count+=1

                    media_entries.append(tweet_data)
                    retweet_average =  retweet_total / tweet_count 
                    like_average =  like_total / tweet_count 
                    replies_average = replies_total / tweet_count 

                user_info = {
                        "username": username,
                        "media_posts": media_entries,
                        "retweet_average" : retweet_average,
                        "like_average" : like_average,
                        "replies_average" : replies_average
                    }
                


                user_tweets_list.append(user_info)
                print(json.dumps(user_info, indent=4))  # Optional for debugging

            else:
                print(f"No timeline data for user {username}")

        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON for user {username}: {e}")
        except Exception as e:
            print(f"Error processing user {username}: {e}")

    return user_tweets_list


def is_user_following_eachother(usernames, headers):
    url = "https://twitter-api45.p.rapidapi.com/checkfollow.php"
    following_list = []

    for user in usernames:
        for follow in usernames:
            if user != follow:
                querystring = {"user": user, "follows": follow}
                
                try:
                    response = requests.get(url, headers=headers, params=querystring)
                    response.raise_for_status()  # raise error for HTTP errors (like 404/500)
                    data = response.json()

                    user_follows = {
                        "user": user,
                        "following": follow,
                        "IsFollowing": data.get("is_follow", False)
                    }
                    following_list.append(user_follows)
                    print(json.dumps(user_follows, indent=4))
                except requests.exceptions.RequestException as e:
                    print(f"Request error for {user} -> {follow}: {e}")
                except ValueError:
                    print(f"JSON decoding error for {user} -> {follow}")

    return following_list

import csv

def save_user_tweets_to_csv(user_tweets, filename='user_tweets.csv'):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        header = ["username", "retweet_average", "like_average", "replies_average", "post_text", "views", "date_created", "replies", "retweets", "likes"]
        writer.writerow(header)

        for user in user_tweets:
            for post in user["media_posts"]:
                writer.writerow([
                    user["username"],
                    user["retweet_average"],
                    user["like_average"],
                    user["replies_average"],
                    post["text"],
                    post["views"],
                    post["date_created"],
                    post["replies"],
                    post["retweets"],
                    post["likes"]
                ])

def save_user_info_to_csv(user_info, filename='user_info.csv'):
    if not user_info:
        return
    keys = user_info[0].keys()
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(user_info)

def save_user_following_to_csv(following_list, filename='user_following.csv'):
    if not following_list:
        return
    keys = following_list[0].keys()
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(following_list)


user_tweets = get_user_tweets(usernames,headers)
save_user_tweets_to_csv(user_tweets)

user_info = get_user_info(usernames,headers)
save_user_info_to_csv(user_info)

user_following = is_user_following_eachother(usernames , headers)
save_user_following_to_csv(user_following)


