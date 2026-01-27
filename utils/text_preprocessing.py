import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords if needed
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Initialize Porter Stemmer
port_stem = PorterStemmer()


def stemming(content):
    
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)  # remove everything except letters
    stemmed_content = stemmed_content.lower()  # convert all text to lowercase
    stemmed_content = stemmed_content.split()  # split text to individual words
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]  # remove stopwords and apply stemming
    stemmed_content = ' '.join(stemmed_content)  # join the list back in one sentence
    return stemmed_content