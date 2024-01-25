# Suicide Ideation Detection using Fasttext Word Embedding and LSTM

This project was originally my undergraduate thesis but then I decided to documented it so that everyone can give it a try too!. You can check the deployed model on [my huggingface here!](https://huggingface.co/apricitea)

The project can be reproduced by using Poetry package manager by building a virtual environment for all the dependencies needed to run the (almost) entire process. The Free Twitter (now, known as X) Developer API access can't query tweets anymore but you can use the final_dataset.xlsx available on this project. For full steps taken by me to complete the project, you can read the post on [my medium here!](https://medium.com/@apricitea) 

Explanation about the folder and files in this repository:
- dataset: contains the final dataset that i use to build the deployed model 
- hunspell-id-main: tools needed to build a hunspell stemmer and stopword removal function
- main: primary documentation of the steps i performed to build the model. 1) crawling twitter data using the twitter_crawl_data.ipnyb, 2) performing stemming and stopwords removal comparison of three different tools (sastrawi, hunspell, and stanza), and 3) perform preprocessing, building word clouds, loading the word embedding matrix, building the text classification model, and evaluating the metrics of each different model-building scenarios.
- IMPORTANT: what my repository doesn't contain: pre-trained fasttext word vectors for indonesian language. you can [find it here](https://fasttext.cc/docs/en/crawl-vectors.html) and download it yourself (Ctrl+F to find "Indonesia" and download the .bin file).
