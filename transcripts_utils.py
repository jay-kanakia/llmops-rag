import pathlib
import re
from pathlib import Path

TIMESTAMP_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}$")

def clean_transcripts(transcript:str)->str:
    """Remove timestamp lines from a transcript and return only the spoken text."""
    lines = transcript.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped_lines = line.strip()

        if not stripped_lines:
            continue

        if TIMESTAMP_PATTERN.match(stripped_lines):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def clean_and_save_transcripts(original_path:Path|str,destination_path:Path|str)->None:

    ORIGINAL_PATH = Path(original_path)
    DESTINATION_PATH = Path(destination_path)

    DESTINATION_PATH.mkdir(parents=True,exist_ok=True)

    transcript_paths = ORIGINAL_PATH.glob(pattern="*txt")

    for transcript_path in transcript_paths:

        # read the text file
        transcript_file = transcript_path.read_text(encoding='utf-8')

        # clean the text file
        cleaned_file = clean_transcripts(transcript_file)

        # filename
        filename = transcript_path.name

        # create the ouput path
        output_path = DESTINATION_PATH/filename

        # write the file content
        output_path.write_text(data=cleaned_file,encoding='utf-8')

if __name__ == "__main__":
    clean_and_save_transcripts(original_path='./transcripts',destination_path='./cleaned_transcripts')
