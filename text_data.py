# I don't care 
from google import genai
import json
from collections import defaultdict
from dotenv import load_dotenv
from collections import Counter
import os
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

client = genai.Client(api_key=GEMINI_API_KEY)


STOP_WORDS = set([
    # Articles
    "a", "an", "the",
    
    # Pronouns
    "i", "me", "my", "mine", "you", "your", "yours", "he", "him", "his",
    "she", "her", "hers", "it", "its", "we", "us", "our", "ours", "they",
    "them", "their", "theirs",
    
    # Prepositions
    "in", "on", "at", "to", "for", "with", "by", "about", "into", "of",
    "from", "up", "down", "over", "under",
    
    # Conjunctions
    "and", "but", "or", "nor", "yet", "so", "because", "although",
    
    # Be verbs
    "am", "is", "are", "was", "were", "be", "been", "being",
    
    # Common verbs
    "do", "does", "did", "have", "has", "had", "can", "could", "will",
    "would", "shall", "should", "may", "might", "must",
    
    # Filler words
    "um", "uh", "like", "well", "basically", "literally", "actually",
    "honestly", "anyway", "right", "just", "stuff", "things", "whatever",
    "yeah", "okay", "sort", "kind", "you know"
])


def read_json_file(filepath):
    file = open(filepath, 'r')
    data = json.load(file)
    file.close()
    return data

def llm_summary(transcription):
    print("Asking Gemini...")
    my_json = transcription
    full_speech_text = my_json['transcription']['full_transcript']
    separator_key = "<SEP>"
    

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"""
            Insummarize what was great in the speech. Be clear and concise, don't use markdown.
            Then using a separator of {separator_key} do all of these things but rather for what can be improved.
            Refer too the speaker as 'you', also don't hallucinate. 
            {full_speech_text}
        """,
    )

    good = response.text.split(separator_key)[0]
    bad = response.text.split(separator_key)[1]
    return {
        "good": good,
        "bad": bad
    }

def llm_vocabulary_analyze(counter):
    print("Asking Gemini about vocabulary")
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"""
            Analyze the vocabulary of the speaker. What are the most common words used?
            What parts of their vocabulary could be improved? What are they doing well?
            Be clear and concise. Don't use markdown. Refer to the speaker as 'you'
            Here is the frequency of words in the speech:
            {counter.most_common(100)}
        """,
    )
    return response.text
    
    
def filter_letters(text):
    return ''.join(filter(str.isalpha, text))

def stripped_length(data):
    offStart = data['transcription']['utterances'][0]['start']
    offEnd = data['transcription']['utterances'][-1]['end']
    return offEnd - offStart


def segment(start, end, value):
    return {
        "start": start,
        "end": end,
        "value": value
    }

def get_wpm_array(data, window_size=10):
    # Create array for word counts
    duration = stripped_length(data)
    start_offset = data['transcription']['utterances'][0]['start']
    num_windows = int(duration // window_size) + 1
    word_counts = [0] * num_windows
    when = [[None, None] for _ in range(num_windows)]
    
    # Count words in each window
    for utterance in data['transcription']['utterances']:
        for word in utterance['words']:
            if filter_letters(word['word'].strip()):  # Only count if word has letters
                length = len(filter_letters(word['word']) ) / 4.5
                window_idx = int((word['start'] - start_offset) // window_size)
                if when[window_idx][0] is None:
                    when[window_idx][0] = word['start']
                when[window_idx][1] = word['end']
                word_counts[window_idx] += length
    
    # Convert to WPM 
    divisor = [60 / (when[i][1] - when[i][0]) for i in range(num_windows)]
    wpms = [int(count * divisor[i]) for i, count in enumerate(word_counts)]
    return [segment(when[i][0], when[i][1], wpms[i]) for i in range(num_windows)]

def text_data(transcription = None):
    if not transcription:
        transcription = read_json_file("./transcribed.json")
    sentences = transcription['transcription']['sentences']
    sentenceSegments = []
    pauseLengths = []

    letterCount = 0

    previousSentence = None
    
    for sentence in sentences: 
        if previousSentence:
            pause_length = sentence['start'] - previousSentence['end']
            pauseLengths.append(segment(previousSentence['end'], sentence['start'],round(pause_length, 1) ))
        sentenceSegments .append(segment(sentence['start'], sentence['end'], sentence['sentence']))
        # get the pause lengths 
        previousSentence = sentence
        letterCount += len(filter_letters(sentence['sentence']))
    
    average_wpm = sum([x['value'] for x in get_wpm_array(transcription)]) / len(get_wpm_array(transcription))
    full_transcript = transcription['transcription']['full_transcript']
    word_frequency = Counter([word for word in full_transcript.split() if word.lower() not in STOP_WORDS])
    print(word_frequency.most_common(10))

    dataObject = {
        "sentences": sentenceSegments,
        "shortest": min(sentenceSegments, key=lambda x: len(x['value'])),
        "longest": max(sentenceSegments, key=lambda x: len(x['value'])),
        "pauseLengths": pauseLengths,
        "wordsPerMinute": get_wpm_array(transcription),
        "summary": llm_summary(transcription),
        "averageWPM": average_wpm,
        "averagePause": sum([x['value'] for x in pauseLengths]) / len(pauseLengths),
        "entireSpeech": full_transcript,
        "wordFrequency": word_frequency,
        "vocabularyAnalysis": llm_vocabulary_analyze(word_frequency),
    }
    return dataObject



if __name__ == '__main__':
    print(text_data())
