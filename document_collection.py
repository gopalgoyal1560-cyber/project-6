from unstructured.partition.auto import partition
from unstructured.cleaners.core import replace_unicode_quotes
from unstructured.cleaners.core import clean
from unstructured.cleaners.core import replace_unicode_quotes
from unstructured.cleaners.core import group_broken_paragraphs
def document_partition(files:list):
    if files:
        elements = []
        for file in files:
            elements.append(partition(filename=file))

        return elements

def document_cleaning(Text:list):
    text = replace_unicode_quotes(Text)
    text = clean(text,extra_whitespace=True)
    text = replace_unicode_quotes(text)
    text = group_broken_paragraphs(text)
    return text


test1 = document_partition(["C:\\Users\\gopal\\OneDrive\\Desktop\\python\\hunter\\index.html"])
print(test1[0][0].text)
test1[0][0].text = document_cleaning(test1[0][0].text)
print(test1[0][0].text)
