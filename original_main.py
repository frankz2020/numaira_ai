import os
# import time
import asyncio
os.environ["TOKENIZERS_PARALLELISM"] = "false"


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer
from RAG import format_maps, find_changes
from funnels import read_docx, excel_to_list, selection

model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
file_path = 'test.docx'
filename = 'test_new_rag.xlsx'

# initialing
excel_value = excel_to_list(filename)
sentences = read_docx(file_path)

changed_sentences = {}
threshold = 0.5

# find_changes
changed_sentences, total_find_sentences_time, total_input_change_time = find_changes(excel_value, sentences, model,
                                                                                     threshold)
    
# filtering irrelevant sentences
changed_sentences = selection(changed_sentences, sentences)
# format_maps
sentences=asyncio.run(format_maps(changed_sentences, sentences))

for key in changed_sentences:
    print(sentences[key])
    
# 画一个饼状图
# import matplotlib.pyplot as plt
#
# labels = ['initialing_time', 'find_changes_time', 'changed_sentences_time']
# sizes = [initialing_time / total_time, find_changes_time / total_time, changed_sentences_time / total_time]
# colors = ['yellowgreen', 'lightskyblue', 'lightcoral']
# plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
# plt.axis('equal')
# plt.show()
#
# labels = ['one_find_sentences_time', 'one_input_change_time']
# sizes = [total_find_sentences_time / find_changes_time, total_input_change_time / find_changes_time]
# colors = ['lightskyblue', 'lightcoral']
# plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
# plt.axis('equal')
# plt.show()
