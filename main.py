from sentence_transformers import SentenceTransformer
from format_mapping import format_maps
from document_processing import excel_to_list, read_docx
from similarity import find_changes

model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
file_path = 'test.docx'
filename = 'test_new_rag.xlsx'  # Replace with your Excel file path

#initialing
excel_value = excel_to_list(filename)
sentences = read_docx(file_path)
changed_sentences = {}
threshold=0.5

changed_sentences = find_changes(excel_value, sentences, model, threshold)

for key, values in changed_sentences.items():
    temp = sentences[key]
    for value in values:
        temp = format_maps(temp, value[0],value[1])
    sentences[key] = temp

for key in changed_sentences:
    print(sentences[key])