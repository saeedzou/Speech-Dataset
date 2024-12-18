import eng_to_ipa as ipa
import nltk
from nltk.corpus import cmudict
from parsivar import Normalizer
import argparse

nltk.download("cmudict")

def parse_args():
    parser = argparse.ArgumentParser(description="Transliterate English words to Persian")
    parser.add_argument("word", type=str, help="input word to transliterate")
    return parser.parse_args()

# Map English IPA sequences to Pinglish equivalents
IPA_TO_PINGLISH = {
    "aɪ": "ai", "ɔɪ": "oy", "ʌ": "a", "oʊ": "o", "eɪ": "ey", "ɪŋ": "ing", "ɑː": "a", "æ": "æ", "ɑ":'a',
    "e": "e", "i": "i", "o": "o", "ɒ": "a", "ɜː": "er", "ə": "e", "ɜ": "e", "ʊ": "u", "aʊ":"ow",
    "ɔː": "o", "uː": "u", "ɪ": "i", "ɛ": "e", "ɔ":"a",
    "l": "l", "v": "v", "p": "p", "r": "r",
    "g": "g", "m": "m", "h": "h", "ŋg": "ng",
    "w": "v", "j": "y", "u": "u", "tʃ": "ch", "dʒ": "j",
    "ʃ": "sh", "ʒ": "zh", "θ": "th", "ð": "th", "ŋ": "ng",
    "z": "z", "t": "t", "s": "s", "k": "k", "d": "d",
    "f": "f", "b": "b", "n": "n",
}

# Map Pinglish to Persian equivalents
PINGLISH_TO_PERSIAN = {
    "a": "ا", "l": "ل", "v": "و", "p": "پ", "r": "ر",
    "o": "و", "g": "گ", "m": "م", "h": "ه", "ei": "ای", "ing": "ینگ",
    "ch": "چ", "sh": "ش", "zh": "ژ", "th": "ث", "j": "ج", "ng": "نگ",
    "z": "ز", "t": "ت", "k": "ک", "d": "د",
    "f": "ف", "b": "ب", "n": "ن", "er": "ار", "e": "ی", "i": "ی",
    "u": "و", "a": "ا", "o": "او", "y": "ی",
    "s_start":'اِس',
    "s_middle_end":"س",
    "ow_middle_end":'و',
    "ow_start":'او',
    # Mapping for 'ae' with two cases
    "ae_start": "اَ",  # For the start of the word
    "ae_middle": "َ",  # For the middle of the word
    "ae_end":"ه",
    # Mapping for 'o' with two cases
    "o_start": "اُ",   # For the start of the word
    "o_middle": "ُ",   # For the middle of the word
    "o_end": "و",
    # Mapping for 'e' with two cases
    "e_start": "اِ",   # For the start of the word
    "e_middle": "ِ",   # For the middle of the word
    "e_end": "ه",
    # Mapping for "i"
    "i_start": "ای",   # For the start of the word
    "i_middle_end": "ی", # For the middle or end of the word
    # Mapping for "ai"
    "ai_start": "آی",  # For the start of the word
    "ai_middle_end": "ای", # For the middle of the word
}
# ARPAbet to IPA mapping
ARPABET_TO_IPA = {
    # Vowels - Monophthongs
    'AO': 'ɔ', 'AO0': 'ɔ', 'AO1': 'ɔ', 'AO2': 'ɔ',
    'AA': 'ɑ', 'AA0': 'ɑ', 'AA1': 'ɑ', 'AA2': 'ɑ',
    'IY': 'i', 'IY0': 'i', 'IY1': 'i', 'IY2': 'i',
    'UW': 'u', 'UW0': 'u', 'UW1': 'u', 'UW2': 'u',
    'EH': 'e', 'EH0': 'e', 'EH1': 'e', 'EH2': 'e',
    'IH': 'ɪ', 'IH0': 'ɪ', 'IH1': 'ɪ', 'IH2': 'ɪ',
    'UH': 'ʊ', 'UH0': 'ʊ', 'UH1': 'ʊ', 'UH2': 'ʊ',
    'AH': 'ʌ', 'AH0': 'ə', 'AH1': 'ʌ', 'AH2': 'ʌ',
    'AE': 'æ', 'AE0': 'æ', 'AE1': 'æ', 'AE2': 'æ',
    'AX': 'ə', 'AX0': 'ə', 'AX1': 'ə', 'AX2': 'ə',

    # Vowels - Diphthongs
    'EY': 'eɪ', 'EY0': 'eɪ', 'EY1': 'eɪ', 'EY2': 'eɪ',
    'AY': 'aɪ', 'AY0': 'aɪ', 'AY1': 'aɪ', 'AY2': 'aɪ',
    'OW': 'oʊ', 'OW0': 'oʊ', 'OW1': 'oʊ', 'OW2': 'oʊ',
    'AW': 'aʊ', 'AW0': 'aʊ', 'AW1': 'aʊ', 'AW2': 'aʊ',
    'OY': 'ɔɪ', 'OY0': 'ɔɪ', 'OY1': 'ɔɪ', 'OY2': 'ɔɪ',

    # Consonants - Stops
    'P': 'p', 'B': 'b', 'T': 't', 'D': 'd', 'K': 'k', 'G': 'g',

    # Consonants - Affricates
    'CH': 'tʃ', 'JH': 'dʒ',

    # Consonants - Fricatives
    'F': 'f', 'V': 'v', 'TH': 'θ', 'DH': 'ð', 'S': 's', 'Z': 'z',
    'SH': 'ʃ', 'ZH': 'ʒ', 'HH': 'h',

    # Consonants - Nasals
    'M': 'm', 'N': 'n', 'NG': 'ŋ',

    # Consonants - Liquids
    'L': 'l', 'R': 'r',

    # Vowels - R-colored vowels
    'ER': 'ɜr', 'ER0': 'ɜr', 'ER1': 'ɜr', 'ER2': 'ɜr',
    'AXR': 'ər', 'AXR0': 'ər', 'AXR1': 'ər', 'AXR2': 'ər',

    # Consonants - Semivowels
    'W': 'w', 'Y': 'j',
}


# Convert ARPAbet transcription to IPA
def arpabet_to_ipa_conversion(arpabet):
    ipa_transcription = ""
    for symbol in arpabet:
        ipa_transcription += ARPABET_TO_IPA.get(symbol, symbol)  # Default to symbol if not found
    return ipa_transcription


# Function to convert word to IPA using CMU Pronouncing Dictionary and ARPAbet to IPA conversion
def word_to_ipa(word, d=cmudict.dict()):
    # Get ARPAbet transcription from CMU dictionary
    word = word.lower()
    if word in d:
        arpabet_transcription = d[word][0]  # Take the first pronunciation
        ipa_transcription = arpabet_to_ipa_conversion(arpabet_transcription)
        return ipa_transcription
    else:
        return None


def clean_ipa(ipa_text):
    """Remove stress markers and extraneous symbols from IPA."""
    stress_markers = ['ˈ', 'ˌ']
    for marker in stress_markers:
        ipa_text = ipa_text.replace(marker, "")
    return ipa_text

def ipa_to_pinglish_conversion(ipa_text):
    """Convert IPA sequences to Pinglish."""
    pinglish = ""
    i = 0
    while i < len(ipa_text):
        match = None
        # Try to match the longest IPA pattern from the current position
        for ipa_seq in sorted(IPA_TO_PINGLISH.keys(), key=len, reverse=True):
            if ipa_text[i:i+len(ipa_seq)] == ipa_seq:
                match = ipa_seq
                break
        if match:
            pinglish += IPA_TO_PINGLISH[match]
            i += len(match)  # Move forward by the length of the match
        else:
            pinglish += ipa_text[i]  # Preserve unmatched characters
            i += 1
    return pinglish


def pinglish_to_persian_conversion(pinglish_text):
    """Convert Pinglish to Persian."""
    persian = ""
    i = 0
    while i < len(pinglish_text):
        length = 1
        match = None
        # Check for 'ae', 'o', or 'e' at the start, middle, or end of a word
        if pinglish_text[i:i+1] == "æ":
            # Check position: Start, Middle, or End of the word
            if i == 0:
                match = "ae_start"
            elif i == len(pinglish_text) - 1:
                match = "ae_end"
            else:
                match = "ae_middle"
        elif pinglish_text[i:i+2] == "ai":
            length = 2
            # Check position: Start, Middle, or End of the word
            if i == 0:
                match = "ai_start"
            else:
                match = "ai_middle_end"
        elif pinglish_text[i:i+2] == "ow":
            length = 2
            # Check position: Start, Middle, or End of the word
            if i == 0:
                match = "ow_start"
            else:
                match = "ow_middle_end"
        elif pinglish_text[i:i+2] == "sh":
            length = 2
            # Check position: Start, Middle, or End of the word
            match = "sh"
        elif pinglish_text[i:i+1] == "o":
            # Check position: Start, Middle, or End of the word
            if i == 0:
                match = "o_start"
            elif i == len(pinglish_text) - 1:
                match = "o_end"
            else:
                match = "o_middle"
        elif pinglish_text[i:i+1] == "e":
            # Check position: Start, Middle, or End of the word
            if i == 0:
                match = "e_start"
            elif i == len(pinglish_text) - 1:
                match = "e_end"
            else:
                match = "e_middle"
        elif pinglish_text[i:i+1] == "i":
            # Check position: Start, Middle, or End of the word
            if i == 0:
                match = "i_start"
            else:
                match = "i_middle_end"
        elif pinglish_text[i:i+1] == "s":
            # Check position: Start, Middle, or End of the word
            if i == 0 and pinglish_text[i+1] not in "aeoui":
                match = "s_start"
            else:
                match = "s_middle_end"
        
        # If a match is found, add the corresponding Persian equivalent
        if match:
            persian += PINGLISH_TO_PERSIAN[match]
            i += length  # Move forward by the length of the match
        else:
            # Try to match the longest Pinglish pattern
            for pinglish_seq in sorted(PINGLISH_TO_PERSIAN.keys(), key=len, reverse=True):
                if pinglish_text[i:i+len(pinglish_seq)] == pinglish_seq:
                    match = pinglish_seq
                    break
            if match:
                persian += PINGLISH_TO_PERSIAN[match]
                i += len(match)
            else:
                persian += pinglish_text[i]  # Preserve unmatched characters
                i += 1
    return persian


def convert_to_persian(english_text, normalizer):
    """Convert English text to Persian pronunciation."""
    # Convert English text to IPA
    ipa_text = word_to_ipa(english_text)
    # Clean the IPA transcription
    if ipa_text:
        cleaned_ipa = clean_ipa(ipa_text)
        pinglish_text = ipa_to_pinglish_conversion(cleaned_ipa)
        persian_text = pinglish_to_persian_conversion(pinglish_text)
    else:
      persian_text = normalizer.normalize(english_text)
    
    return persian_text

def main():
    args = parse_args()
    assert len(args.word.split()) == 1
    normalizer = Normalizer(pinglish_conversion_needed=True)
    print(args.word)
    print(convert_to_persian(args.word, normalizer))

if __name__ == '__main__':
    main()
