# I don't care 
key = "AIzaSyD2hGhPlGUU_1mLnN6Co6WuDPAbBlIOB8A"
from google import genai
import json
from collections import defaultdict

client = genai.Client(api_key=key)


def read_json_file(filepath):
    file = open(filepath, 'r')
    data = json.load(file)
    file.close()
    return data

def llm_summary():
    print("Asking Gemini...")
    my_json = read_json_file("./transcribed.json")
    full_speech_text = my_json['transcription']['full_transcript']
    separator_key = "<SEP>"
    

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"""
            In max 50 words, summarize what was great in the speech. Be clear and concise.
            Then using a separator of {separator_key} do the same thing but for what could be improved.
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
    
    
def filter_letters(text):
    return ''.join(filter(str.isalpha, text))

def stripped_length(data):
    offStart = data['transcription']['utterances'][0]['start']
    offEnd = data['transcription']['utterances'][-1]['end']
    return offEnd - offStart

def get_wpm_array(data, window_size=10):
    # Create array for word counts
    duration = stripped_length(data)
    start_offset = data['transcription']['utterances'][0]['start']
    num_windows = int(duration // window_size) + 1
    word_counts = [0] * num_windows
    
    # Count words in each window
    for utterance in data['transcription']['utterances']:
        for word in utterance['words']:
            if filter_letters(word['word'].strip()):  # Only count if word has letters
                length = len(filter_letters(word['word']) ) / 4.5
                window_idx = int((word['start'] - start_offset) // window_size)
                word_counts[window_idx] += length
    
    # Convert to WPM (multiply by 6 since windows are 10 seconds)
    return [int(count * 6) for count in word_counts]

def text_data():
    data = read_json_file("./transcribed.json")
    sentences = data['transcription']['utterances']
    sentenceAndLength = []
    pauseLengths = []

    letterCount = 0

    previousSentence = None
    
    for sentence in sentences: 
        if previousSentence:
            pauseLengths.append(sentence['start'] - previousSentence['end'])
        length = sentence['end'] - sentence['start']
        sentenceAndLength.append((sentence['text'], length))
        # get the pause lengths 
        previousSentence = sentence
        letterCount += len(filter_letters(sentence['text']))

    dataObject = {
        "sentences": sentenceAndLength,
        "shortest": min(sentenceAndLength, key=lambda x: x[1]),
        "longest": max(sentenceAndLength, key=lambda x: x[1]),
        "pauseLengths": pauseLengths,
        "wordsPerMinute": get_wpm_array(data),
        "summary": llm_summary()
    }
    return dataObject



if __name__ == '__main__':
    print(text_data())
