import csv
#import snscrape.modules.twitter as sntwitter
import tweepy

MAX_TWEETS = 20

def get_tweets():
    # Snscrape
    """
    tweets = []
    query = 'javier milei'  
    since_date = '2023-01-01'

    id = 0
    for tweet in sntwitter.TwitterSearchScraper(f'{query} since:{since_date} lang:en').get_items():
        tweets.append({
            'ids': id,
            'date': tweet.date.strftime('%Y-%m-%d'),
            'flag': query if query else 'NO QUERY',
            'text': tweet.content,       
        })
        id += 1
        
        print(tweets[id - 1])

        if len(tweets) >= MAX_TWEETS:
            break
    """

    #Tweepy
    consumer_key = "wNhENHAMuuuMcSgOnGWfkC3j1"
    consumer_secret = "TsSNltReH2cQcp4PeOQYLk4gD39a06rhrXpZzSqRwlWk21JFzQ"
    access_token = "1671943750396903433-rLfaZVLPaEc57cO4hP16CWWwvBRITI"
    access_token_secret = "jjXJwzmPI2eZlw6slBD5vlyQ4NJgGspiP5R24XGK0grcI"

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    query = 'Javier Milei'
    num_tweets = 3
    caba_geocode = "-34.611781,-58.417309, 100km"

    tweets = api.search_tweets(q = query, geocode = caba_geocode, result_type = "mixed", lang = 'es').items(num_tweets)

    for tweet in tweets:
        print(f'Tweet by {tweet.user.screen_name}: {tweet.text}')


    return tweets

if __name__ == "__main__":
    get_tweets()
