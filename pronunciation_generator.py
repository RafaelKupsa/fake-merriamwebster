import openai

from gpt import GPT
from gpt import Example

import config

class PronunciationGenerator:
    """
    class for generating a pronunciation string from any word
    """
    def __init__(self):
        """
        Initializes the generator
        """

        # Reading the API key from a file
        openai.api_key = config.api_key

        # 30 shots
        self.gpt = GPT(engine="davinci", temperature=0.5, max_tokens=100)
        self.gpt.add_example(Example('winsome', 'WIN-sum'))
        self.gpt.add_example(Example('stola', 'STOH-luh'))
        self.gpt.add_example(Example('sanguine', 'SANG-gwin'))
        self.gpt.add_example(Example('euphemism', 'YOO-fuh-miz-um'))
        self.gpt.add_example(Example('gloss', 'GLAHSS'))
        self.gpt.add_example(Example('meritorious', 'mair-uh-TOR-ee-us'))
        self.gpt.add_example(Example('stir-crazy', 'STER-KRAY-zee'))
        self.gpt.add_example(Example('tome', 'TOHM'))
        self.gpt.add_example(Example('affable', 'AF-uh-bul'))
        self.gpt.add_example(Example('finesse', 'fuh-NESS'))
        self.gpt.add_example(Example('voluble', 'VAHL-yuh-bul'))
        self.gpt.add_example(Example('layman', 'LAY-mun'))
        self.gpt.add_example(Example('cerulean', 'suh-ROO-lee-un'))
        self.gpt.add_example(Example('antithetical', 'an-tuh-THET-ih-kul'))
        self.gpt.add_example(Example('palindrome', 'PAL-un-drohm'))
        self.gpt.add_example(Example('captious', 'KAP-shuss'))
        self.gpt.add_example(Example('rejuvenate', 'rih-JOO-vuh-nayt'))
        self.gpt.add_example(Example('zeitgeist', 'TSYTE-gyste'))
        self.gpt.add_example(Example('astute', 'uh-STOOT'))
        self.gpt.add_example(Example('carte blanche', 'KART-BLAHNCH'))  # 20
        self.gpt.add_example(Example('quip', 'KWIP'))
        self.gpt.add_example(Example('intemperate', 'in-TEM-puh-rut'))
        self.gpt.add_example(Example('juggernaut', 'JUG-er-nawt'))
        self.gpt.add_example(Example('opine', 'oh-PYNE'))
        self.gpt.add_example(Example('derring-do', 'dair-ing-DOO'))
        self.gpt.add_example(Example('non sequitur', 'NAHN-SEK-wuh-ter'))
        self.gpt.add_example(Example('debilitating', 'dih-BILL-uh-tay-ting'))
        self.gpt.add_example(Example('insinuate', 'in-SIN-yuh-wayt'))
        self.gpt.add_example(Example('livid', 'LIV-id'))
        self.gpt.add_example(Example('Kwanzaa', 'KWAHN-zuh'))

    def generate(self, word):
        """
        Generates the pronunciation for the given word
        Params:
            word: the word as a string
        Returns:
            the pronunciation as a string
        """
        return self.gpt.submit_request(word).choices[0].text.replace("output:", "").strip().split()[0].strip()
