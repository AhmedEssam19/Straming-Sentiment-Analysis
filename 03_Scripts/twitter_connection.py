from tweepy import Stream
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
import socket
import json

consumer_key = 'ciYOQNxLfrKbnlehMaTg5R3Od'
consumer_secret = 'Pe4AKi84Qvh2w6EOVdDtAwzEugh1LSABaTfzzXHEzEp0xchcii'
access_token = '2993081723-r7YbLmTnVmFa3M5dNluzFlDb5Kp4yivbEtNKKOG'
access_secret = 'OQNvW3M0XYQKMi1C1O2x1QOCEkjJdDerRPKyCQUP14hgP'


class TweetsListener(StreamListener):
    # tweet object listens for the tweets
    def __init__(self, client_socket):
        super().__init__()
        self.client_socket = client_socket

    def on_data(self, data):
        try:
            msg = json.loads(data)
            print("new message")
            # if tweet is longer than 140 characters
            if "extended_tweet" in msg:
                # add at the end of each tweet "t_end"
                self.client_socket.send(str(msg['extended_tweet']['full_text'] + "t_end").encode('utf-8'))
            else:
                # add at the end of each tweet "t_end"
                self.client_socket.send(str(msg['text'] + "t_end").encode('utf-8'))
            return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True

    def on_error(self, status):
        print(status)
        return True


def sendData(client_socket, keyword):
    print('start sending data from Twitter to socket')
    # authentication based on the credentials
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    # start sending data from the Streaming API
    twitter_stream = Stream(auth, TweetsListener(client_socket))
    twitter_stream.filter(track=keyword, languages=["en"])


if __name__ == "__main__":
    # get th keyword
    with open('keyword.txt') as file:
        keyword = file.read().strip()
    # server (local machine) creates listening socket
    s = socket.socket()
    host = "0.0.0.0"
    port = 5557
    s.bind((host, port))
    print('socket is ready')
    # server (local machine) listens for connections
    s.listen(4)
    print('socket is listening')
    # return the socket and the address on the other side of the connection (client side)
    c_socket, address = s.accept()
    print("Received request from: " + str(address))
    # select here the keyword for the tweet data
    sendData(c_socket, keyword=[keyword])
