from twitter_keys import *
import tweepy
import os
import time
from datetime import date, timedelta
from word_generator import WordGenerator
from meaning_generator import MeaningGenerator
from pronunciation_generator import PronunciationGenerator
from image_generator import ImageGenerator


def setup_api():
    """
    Sets up the twitter API
    Returns:
        an authorized tweepy.API object
    """
    auth = tweepy.OAuthHandler(api_key, api_secret)
    auth.set_access_token(access_token, access_secret)
    return tweepy.API(auth)


def generate(wg, pg, mg, ig, post_date=None):
    """
    Generates a #FakeWordOfTheDay tweet and image
    Params:
        wg: a WordGenerator instance
        pg: a PronunciationGenerator instance
        mg: a MeaningGenerator instance
        ig: an ImageGenerator instance
        post_date: date on which the tweet will be posted as a datetime.date
    Returns:
        tweet: the tweet with the word and the example sentence
        image: the image as a PIL.Image object
    """
    word, pos, _, affix_meanings = wg.generate()
    pronunciation = pg.generate(word)
    print(pronunciation)
    meaning, example = mg.generate(word, pos, affix_meanings)
    image = ig.generate(word, pos, pronunciation, meaning, post_date)
    tweet = f"Good Morning! Today's #FakeWordOfTheDay is '{word}'. Example sentence: {example}"
    while len(tweet) > 280:
        word, pos, _, affix_meanings = wg.generate()
        pronunciation = pg.generate(word)
        meaning, example = mg.generate(word, pos, affix_meanings)
        image = ig.generate(word, pos, pronunciation, meaning)
        tweet = f"Good Morning! Today's #FakeWordOfTheDay is '{word}'. Example sentence: {example}"
    return tweet, image


def generate_in_advance(days):
    """
    Generates tweets and images in adavance for the given number of days
    Tweets are saved in a file called "tweets" (The date and one tweet per line, separated by "\t")
    Image are saved in the folder "image_samples" with the date as the file name.
    Params:
        days: number of days as an int
    """
    wg = WordGenerator()
    pg = PronunciationGenerator()
    mg = MeaningGenerator()
    ig = ImageGenerator()

    for i in range(days):
        post_date = date.today() + timedelta(days=i)
        tweet, image = generate(wg, pg, mg, ig, post_date)
        date_formatted = post_date.strftime("%B %d, %Y").upper()

        # Save image
        image.save(os.path.join("image_samples", date_formatted + ".png"))

        # Save tweet
        with open("tweets", "a") as f:
            f.write(f"{date_formatted}\t{tweet}\n")


def main():
    """
    Generates an image and posts it on twitter
    """
    api = setup_api()
    api.verify_credentials()
    wg = WordGenerator()
    pg = PronunciationGenerator()
    mg = MeaningGenerator()
    ig = ImageGenerator()
    tweet, image = generate(wg, pg, mg, ig)
    fp = "temp.png"
    image.save(fp)
    print(tweet)
    media = api.media_upload(filename=fp)
    api.update_status(status=tweet, media_ids=[media.media_id])
    os.remove(fp)

    return "Tweet Posted"


if __name__ == "__main__":
    generate_in_advance(30)