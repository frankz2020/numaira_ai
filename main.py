import os
import time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer
from format_mapping import format_maps
from document_processing import excel_to_list, read_docx
from similarity import find_changes

model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
file_path = 'test.docx'
filename = 'test_new_rag.xlsx'

#initialing
start_time=time.time()
initialing_time=time.time()
excel_value = excel_to_list(filename)
sentences = read_docx(file_path)
initialing_time=time.time()-initialing_time

changed_sentences = {}
threshold=0.5

# find_changes
find_changes_time=time.time()
changed_sentences,total_find_sentences_time,total_input_change_time = find_changes(excel_value, sentences, model, threshold)
find_changes_time=time.time()-find_changes_time

# format_maps
changed_sentences_time=time.time()
format_maps(changed_sentences,sentences)
changed_sentences_time=time.time()-changed_sentences_time
end_time=time.time()


#time calculation
total_time=end_time-start_time
print("total_time: "+str(total_time))
print("initialing_time: "+str(initialing_time))
print("find_changes_time: "+str(find_changes_time))
print("changed_sentences_time: "+str(changed_sentences_time))
print('/n/n/n')
print("initialing_time: "+str(initialing_time/total_time))
print("find_changes_time: "+str(find_changes_time/total_time))
print("changed_sentences_time: "+str(changed_sentences_time/total_time))

for key in changed_sentences:
    print(sentences[key])
#画一个饼状图
import matplotlib.pyplot as plt
labels = ['initialing_time', 'find_changes_time', 'changed_sentences_time']
sizes = [initialing_time/total_time, find_changes_time/total_time, changed_sentences_time/total_time]
colors = ['yellowgreen', 'lightskyblue', 'lightcoral']
plt.pie(sizes,labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.show()

labels = ['one_find_sentences_time', 'one_input_change_time']
sizes = [total_find_sentences_time/find_changes_time, total_input_change_time/find_changes_time]
colors = ['lightskyblue', 'lightcoral']
plt.pie(sizes,labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.show()

