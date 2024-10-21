import os
from pypdf import PdfReader
import pathlib


def extract_text(pdf):
    print(f'Extracting text from {pdf}')
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    reader.close()
    return text


def extract_pdfs(source, destination):
    base_dir = pathlib.Path(__file__).parent.resolve().parent
    os.makedirs(f'{base_dir}/{destination}', exist_ok=True)
    for pdf in os.listdir(f'{base_dir}/{source}'):
        text = extract_text(f'{base_dir}/{source}{pdf}')
        with open(f'{base_dir}/{destination}/{pdf}.txt', 'w') as f:
            f.write(text)


if __name__ == '__main__':
    extract_pdfs('pdf/', 'doc/')
