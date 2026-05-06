import dspy
import os


def msg_to_intent_metric(example, pred, trace=None):
    return pred.intent_label == example.intent_label


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
lm = dspy.LM("openai/gpt-4o-mini", api_key=OPENAI_API_KEY)
dspy.settings.configure(lm=lm)
labels = ['Cancel Subscription', 'Update Email', 'Refund Request',
          'Bug Report', 'Unknown']
examples = [dspy.Example({'message': 'I want to end my subscription',
                          'labels': labels,
                          'intent_label': 'Cancel Subscription'},
                         )
            .with_inputs('message'),
            dspy.Example({'message': 'I deserve a refund',
                          'labels': labels,
                          'intent_label': 'Refund Request'}
                         ),
            dspy.Example({'message': 'The login screen is frozen, I cant login',
                          'labels': labels,
                          'intent_label': 'Bug Report'}
                         )
            ]
classifier = dspy.ChainOfThought('message, labels -> intent_label')
optimizer = dspy.BootstrapFewShot(metric=msg_to_intent_metric, max_labeled_demos=2)
classifier = optimizer.compile(classifier, trainset=examples)
prediction = classifier(
    message="My new email address is bob.smith@domain.com", labels=labels)
print(prediction)
