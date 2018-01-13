# encoding: utf-8

import io

input_filename = 'truncated_socialmedia-disaster-tweets.csv'
output_filename = "socialmedia-disaster-tweets_clean.csv"

raw = io.open(input_filename, "r", encoding="utf-8", errors='replace')
clean = open(output_filename, "w")

for line in raw:
    clean.write(line.encode("utf-8"))
raw.close(); clean.close()
