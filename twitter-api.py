import twitter
import re

api = twitter.Api(
    consumer_key='',
    consumer_secret='',
    access_token_key='',
    access_token_secret=''
)

max_id = ''
with open('data/meduzaproject.txt', 'w', encoding='utf-8') as file:
    while(True):
        print(max_id)
        statuses = api.GetUserTimeline(screen_name='meduzaproject', include_rts='false', count=200, max_id=max_id)
        if(len(statuses) > 0):
            print("Number of Tweets: " + str(len(statuses)))
            for s in statuses:
                result = re.sub(r"http\S+", "", s.text.lower())
                file.write(result)
                max_id = s.id - 1
        else:
            break