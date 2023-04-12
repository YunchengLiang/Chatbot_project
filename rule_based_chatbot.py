import random
from nltk import word_tokenize
greetings=['Hi','hi','hey',"what's up", 'wassup','Hey','hello','Hello']

questions=["holiday","vacation","weekend"]
responses=["it was okay","it was alright", "amazing is how i describe it", "not much, how about you?"]

while True:
    userInput= input(">>>>")
    clean_input= word_tokenize(userInput)
    if not set(clean_input).isdisjoint(questions):
        print(random.SystemRandom().choice(responses))
    elif not set(clean_input).isdisjoint(greetings):
        print(random.SystemRandom().choice(greetings)
)
    elif userInput=="bye":
        break
    else:
        print("I did not understand what you said")