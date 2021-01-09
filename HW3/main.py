import os
from nltk.stem.porter import *
import re
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import argparse
import datetime


stopwords = ["a","about","above","after","again","against","ain","all","am","an","and","any","are","aren","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can","couldn","couldn't","d","did","didn","didn't","do","does","doesn","doesn't","doing","don","don't","down","during","each","few","for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn","isn't","it","it's","its","itself","just","ll","m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't","no","nor","not","now","o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's","should","should've","shouldn","shouldn't","so","some","such","t","than","that","that'll","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what","when","where","which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","could","he'd","he'll","he's","here's","how's","i'd","i'll","i'm","i've","let's","ought","she'd","she'll","that's","there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've","what's","when's","where's","who's","why's","would","able","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","afterwards","ah","almost","alone","along","already","also","although","always","among","amongst","announce","another","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","arent","arise","around","aside","ask","asking","auth","available","away","awfully","b","back","became","become","becomes","becoming","beforehand","begin","beginning","beginnings","begins","behind","believe","beside","besides","beyond","biol","brief","briefly","c","ca","came","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","couldnt","date","different","done","downwards","due","e","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","ff","fifth","first","five","fix","followed","following","follows","former","formerly","forth","found","four","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed","hence","hereafter","hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie","im","immediate","immediately","importance","important","inc","indeed","index","information","instead","invention","inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","moreover","mostly","mr","mrs","much","mug","must","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted","nothing","nowhere","obtain","obtained","obviously","often","oh","ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside","overall","owing","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","said","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","shed","shes","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","sufficiently","suggest","sup","sure","take","taken","taking","tell","tends","th","thank","thanks","thanx","thats","that've","thence","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","theyd","theyre","think","thou","though","thoughh","thousand","throug","throughout","thru","thus","til","tip","together","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","unfortunately","unless","unlike","unlikely","unto","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","via","viz","vol","vols","vs","w","want","wants","wasnt","way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole","who'll","whomever","whos","whose","widely","willing","wish","within","without","wont","words","world","wouldnt","www","x","yes","yet","youd","youre","z","zero","a's","ain't","allow","allows","apart","appear","appreciate","appropriate","associated","best","better","c'mon","c's","cant","changes","clearly","concerning","consequently","consider","considering","corresponding","course","currently","definitely","described","despite","entirely","exactly","example","going","greetings","hello","help","hopefully","ignored","inasmuch","indicate","indicated","indicates","inner","insofar","it'd","keep","keeps","novel","presumably","reasonably","second","secondly","sensible","serious","seriously","sure","t's","third","thorough","thoroughly","three","well","wonder"]
stopwords = set(stopwords)
stemmer = PorterStemmer()

feature_size = 230

term_freq = dict()
tfidf = dict()

def tokenization(content):
    content = [re.sub(r"[^a-zA-Z]+", '', word) for word in content]
    content = [word.replace(',', '').replace('.', '').replace('\'', '').replace('\n', '') for word in content]
    content = [stemmer.stem(word) for word in content] # Stemming using Porter’s algorithm
    content = [word for word in content if word not in stopwords and len(word) != 0] # Stopword removal

    return content

def read_documents():
    documents = dict()
    for root, dirs, files in os.walk("IRTM"):
        for file in files:
            if ".txt" not in file:
                continue
            # print(os.path.join(root, file))
            filename = os.path.join(root, file)
            corpus = []
            with open(filename, "r") as f:
                contents = f.readlines()
                for content in contents:
                    text = content.lower()
                    text = text.split(" ")

                    text = tokenization(text)
                    corpus.extend(text)
            # print(file.replace(".txt", ""))
            documents[int(file.replace(".txt", ""))] = corpus
    return documents



def read_all_doc():
    # Calculates the document frequency of each corpus
    doc_freq = dict()
    for root, dirs, files in os.walk("IRTM"):
        for file in files:
            if ".txt" not in file:
                continue
            # print(os.path.join(root, file))
            filename = os.path.join(root, file)
            corpus = set()
            with open(filename, "r") as f:
                contents = f.readlines()
                for content in contents:
                    text = content.lower()
                    text = text.split(" ")

                    text = tokenization(text)
                    corpus.update(text)

            for word in corpus:
                doc_freq[word] = doc_freq.setdefault(word, 0) + 1
    pass

def read_category():

    with open("category.txt", "r") as f:
        lines = f.readlines()
        lines = [line.replace("\n", "").split(" ") for line in lines]
        f.close()
    categories = {}
    for line in lines:
        category = None
        for idx, number in enumerate(line):
            if number == "":
                continue
            # print(idx, number, category)
            if idx == 0:
                category = int(number)
            else:
                categories[int(number)] = category
    return categories


def countDocsInClass(doc_category):
    # Counts number of examples in each category
    # doc_category
    # {
    #     doc_id_0: 0,
    #     doc_id_1: 2,
    #     doc_id_2: 5,
    #     ...
    # }
    # Generates the category prior
    total = 0
    category_occurrence = {k: 0 for k in range(1, 14)}
    for document, category in doc_category.items():
        category_occurrence[category] += 1
        total += 1
    for category in category_occurrence:
        category_occurrence[category] /= total
    return category_occurrence

def collect_all_text(documents, doc_cat):
    # Concatenates all the terms in documents of category "class" into a list.
    class_document = {k: [] for k in range(1, 14)}
    testing_set = set()
    for class_ in range(1, 14):
        for doc_id, document in documents.items():
            if doc_id not in doc_cat:
                # print(doc_id, "not in training set")
                testing_set.add(doc_id)
                continue
            if doc_cat[doc_id] == class_:
                class_document[class_].extend(document)
    return class_document, list(testing_set)

def extractTokens(tokens, document):
    corpus = []
    for text in document:
        if text not in tokens:
            continue
        else:
            corpus.append(text)
    return corpus

def expected_mutual_information(category_full_document, doc_category, documents):
    category_doc = {k: [] for k in range(1, 14)}
    category_selected_terms = {k: set() for k in range(1, 14)}
    epsilon = 1e-9
    
    for doc_id, category in doc_category.items():
        category_doc[category].append(doc_id)
        # {1: [12, 14, 15,...], 2: [13, 17, ...]}
    for class_ in category_full_document:
        print("Selecting class", class_)
        unique_corpus = set(category_full_document[class_])
        score = np.zeros((len(unique_corpus)))
        # calculate number of documents where term "t" is present and category is "class_"
        
            # 15/ 195 * np.log()
        
        for idx, term in enumerate(unique_corpus):

            t_on_topic = 0 # not_t_on_topic = 15 - t_on_topic
            t_off_topic = 0 # not_t_off_topic = 180 - t_off_topic
            for doc_id in category_doc[class_]:
                t_on_topic += term in documents[doc_id]
            for i in range(1, 14):
                if i == class_:
                    continue
                for doc_id in category_doc[i]:
                    t_off_topic += term in documents[doc_id]
            absent_on_topic = 15 - t_on_topic
            absent_off_topic = 180 - t_off_topic
            t_doc_freq = t_on_topic + t_off_topic
            # document frequency of t is t_on_topic + t_off_topic

            score[idx] = t_on_topic / 195 * np.log2((((t_on_topic * 195) + 1) / (t_doc_freq  * 15 ))) + \
                        t_off_topic / 195 * np.log2((((t_off_topic * 195) + 1) / (t_doc_freq * 180))) + \
                        absent_on_topic / 195 * np.log2((((absent_on_topic * 195) + 1)/ ((195 -t_doc_freq)  * 15)) ) + \
                        absent_off_topic / 195 * np.log2((((absent_off_topic * 195) + 1 )/ ((195 -t_doc_freq) *  180)) )
            #p(et, ec) * np.log(p(et, ec) / (p(et) * p(ec)) )
            # et=1, ec=1
            # et=1, ec=0
            # et=0, ec=1
            # et=0, ec=0
            # print(term, score[idx])
        unique_corpus, score = list(zip(*sorted(zip(list(unique_corpus), score), key=lambda x: x[1], reverse=True)))
        # for i in range(100):
        #     print(unique_corpus[i], score[i])
        category_selected_terms[class_].update(unique_corpus[:feature_size])
        # print(category_selected_terms[class_])
        # input()
    return category_selected_terms

def chi_square_feature_selection(category_full_document, doc_category, documents):
    category_doc = {k: [] for k in range(1, 14)}
    category_selected_terms = {k: set() for k in range(1, 14)}
    full_corpus = set()
    for k, value in category_full_document.items():
        full_corpus.update(value)
    
    for doc_id, category in doc_category.items():
        category_doc[category].append(doc_id)
        # {1: [12, 14, 15,...], 2: [13, 17, ...]}
    score = np.zeros((len(full_corpus)))
    for class_ in category_full_document:
        print("Selecting class", class_)
        full_corpus = set()
        for k, value in category_full_document.items():
            full_corpus.update(value)
        # unique_corpus = set(category_full_document[class_])
        
        # calculate number of documents where term "t" is present and category is "class_"
        
            # 15/ 195 * np.log()
        
        for idx, term in enumerate(full_corpus):

            t_on_topic = 0 # not_t_on_topic = 15 - t_on_topic
            t_off_topic = 0 # not_t_off_topic = 180 - t_off_topic
            for doc_id in category_doc[class_]:
                t_on_topic += term in documents[doc_id]
            for i in range(1, 14):
                if i == class_:
                    continue
                for doc_id in category_doc[i]:
                    t_off_topic += term in documents[doc_id]
            
            t_doc_freq = (t_on_topic + t_off_topic)
            
            # document frequency of t is t_on_topic + t_off_topic

            E_11 = 195 * (t_doc_freq * np.log(195/t_doc_freq) / 195) * (1 / 13)
            
            score[idx] += ((t_on_topic * np.log(195/t_doc_freq) - E_11) ** 2) / E_11
            #p(et, ec) * np.log(p(et, ec) / (p(et) * p(ec)) )
            
    full_corpus, score = list(zip(*sorted(zip(list(full_corpus), score), key=lambda x: x[1], reverse=True)))
    for i in range(feature_size):
        print(full_corpus[i], score[i])
    return full_corpus[:feature_size]

if __name__ == "__main__":

    print("reading all documents...")
    
    doc_category = read_category() # doc_category contains all the training set's label.
    documents = read_documents()
    category_full_document, test_set_ids = collect_all_text(documents, doc_category) # category_full_document contains full docs of a category
    category_prior = countDocsInClass(doc_category)

    full_corpus = []
    for k, value in category_full_document.items():
        full_corpus.extend(value)
    # feature selection
    selected_terms = chi_square_feature_selection(category_full_document, doc_category, documents)

    selected_terms = set(selected_terms)
    corpus_position = {value: ind for ind, value in enumerate(selected_terms)}
    # print(corpus_position)
    conditional_prob = np.zeros((feature_size, 13))
    for class_ in range(1, 14):
        for idx, term in enumerate(selected_terms):
            count_term = category_full_document[class_].count(term)
            conditional_prob[idx, class_ - 1] = (count_term + 1) / (len(category_full_document[class_]) + feature_size * 13)

    # Testing
    predictions = {}
    for idx, test_set_id in enumerate(test_set_ids):
        tokens = extractTokens(full_corpus, documents[test_set_id])
        score = np.zeros(13)
        print(idx, test_set_id)
        for class_ in range(1, 14):
            for token in tokens:
                if token not in selected_terms:
                    continue
                score[class_ - 1] += np.log(conditional_prob[corpus_position[token]][class_ - 1])
            score[class_ - 1] += np.log(category_prior[class_])
        # print(score)
        predictions[test_set_id] = np.argmax(score)

    predictions = {k: v for k, v in sorted(predictions.items(), key=lambda x:x[0] )}
    with open("predictions_{}.csv".format(datetime.datetime.now()), "w") as f:
        f.write("Id,Value\n")
        for k, v in predictions.items():
            f.write("{},{}\n".format(k, v+1))
        f.close()
    print(predictions)


    pass
