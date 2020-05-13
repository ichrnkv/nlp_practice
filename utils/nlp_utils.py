import re


def tokenize_text_regex(text, regex, lower=True, min_token_size=1):
    """
    Tokenize text using regular expression
    :param text: text without prepossessing, str
    :param regex: regular expression, re.compile
    :param lower: lowercase text, True ao False
    :param min_token_size: min token size, int
    """

    if lower:
        text = text.lower()
    all_tokens = regex.findall(text)
    return [token for token in all_tokens if len(token) >= min_token_size]


def tokenize_corpus(corpus, tokenizer, **tokenizer_kwargs):
    """
    Tokenize corpus using regular expression
    :param corpus: text corpus
    :param tokenizer: regex based tokenizer
    :param tokenizer_kwargs: tokenizer params
    """
    
    return [tokenizer(text, **tokenizer_kwargs) for text in corpus]


if __name__ == '__main__':
    with open('./../data/IMDB Dataset.csv', 'r') as file:
        full_text = file.read()

        token_re = re.compile(r'[\w\d]+')
        text_tokenized = tokenize_text_regex(text=full_text, regex=token_re, lower=True, min_token_size=5)
        print(text_tokenized[:10])
