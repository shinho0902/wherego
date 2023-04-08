import random
import string
import bcrypt
from urllib import parse
import re
import random

def makeroom():
    rand_str= ""
    for i in range(random.randint(15,25)):
        rand_str += str(random.choice(string.ascii_uppercase + string.digits))
    password = bcrypt.hashpw(rand_str.encode('utf-8'), bcrypt.gensalt()).decode()
    # password = password.replace("/"."")
    url =  parse.quote(password).replace(".","").replace("/","")
    pattern_punctuation = re.compile(r'[^\w\s]')
    url = pattern_punctuation.sub('', url)
    return url