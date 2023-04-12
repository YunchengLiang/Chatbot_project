from chatterbot import ChatBot
bot= ChatBot(
    'Default Response Example Bot',
    storage_adapter= 'chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        {
            'import_path':'chatterbot.logic.BestMatch'
        },
        {
            'import_path':'chatterbot.logic.LowConfidenceAdapter',
            'threshold':0.65,
            'default_response':'I am sorry, but i do not understand'
         }
    ],
    trainer='chatterbot.trainers.ListTrainer'
    )
#add some sampels to train
bot.train(
    [
    'How can i help you?',
    'I want to create a cha bot',
    'Have you read the documentation?',
    'No, I have not',
    'This should help you get started; hettp://chatterbot.rftd.org/en/latest/quickstart.html'
    ]
)

question = "How do I make an omelette?"
print(question)
response= bot.get_response(question)
print("\n",response)

question = "How do I make a chat bot?"
print(question)
response= bot.get_response(question)
print("\n",response)
