import openai
import wave
import pydub

API_KEY = ''

def convert_to_wav(input_file):
    if input_file.endswith('.mp3'):
        audio = pydub.AudioSegment.from_mp3(input_file)
        wav_file = input_file.replace('.mp3', '.wav')
        audio.export(wav_file, format='wav')
        print(f"Converted {input_file} to {wav_file}")
        return wav_file
    return input_file

def transcribe_audio(file_path):
    file_path = convert_to_wav(file_path)
    client = openai.OpenAI(api_key=API_KEY)
    with open(file_path, 'rb') as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return response.text

def translate_to_hindi(text):
    client = openai.OpenAI(api_key=API_KEY)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Translate all inputs to Hindi without modifying the meaning."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

def text_to_speech(text, output_file):
    client = openai.OpenAI(api_key=API_KEY)
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    response.stream_to_file(output_file)
    print(f"Speech audio saved to {output_file}")

if __name__ == "__main__":
    input_file = 'input_audio.mp3'  

    transcription = transcribe_audio(input_file)
    print("Transcription:", transcription)

    translated_text = translate_to_hindi(transcription)
    print("Translated to Hindi:", translated_text)

    text_to_speech(translated_text, 'output_audio.wav')
