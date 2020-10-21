import os
from nltk.stem.porter import *
import re
import numpy as np


stopwords = ["a","about","above","after","again","against","ain","all","am","an","and","any","are","aren","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can","couldn","couldn't","d","did","didn","didn't","do","does","doesn","doesn't","doing","don","don't","down","during","each","few","for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn","isn't","it","it's","its","itself","just","ll","m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't","no","nor","not","now","o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's","should","should've","shouldn","shouldn't","so","some","such","t","than","that","that'll","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what","when","where","which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","could","he'd","he'll","he's","here's","how's","i'd","i'll","i'm","i've","let's","ought","she'd","she'll","that's","there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've","what's","when's","where's","who's","why's","would","able","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","afterwards","ah","almost","alone","along","already","also","although","always","among","amongst","announce","another","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","arent","arise","around","aside","ask","asking","auth","available","away","awfully","b","back","became","become","becomes","becoming","beforehand","begin","beginning","beginnings","begins","behind","believe","beside","besides","beyond","biol","brief","briefly","c","ca","came","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","couldnt","date","different","done","downwards","due","e","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","ff","fifth","first","five","fix","followed","following","follows","former","formerly","forth","found","four","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed","hence","hereafter","hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie","im","immediate","immediately","importance","important","inc","indeed","index","information","instead","invention","inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","moreover","mostly","mr","mrs","much","mug","must","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted","nothing","nowhere","obtain","obtained","obviously","often","oh","ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside","overall","owing","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","said","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","shed","shes","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","sufficiently","suggest","sup","sure","take","taken","taking","tell","tends","th","thank","thanks","thanx","thats","that've","thence","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","theyd","theyre","think","thou","though","thoughh","thousand","throug","throughout","thru","thus","til","tip","together","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","unfortunately","unless","unlike","unlikely","unto","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","via","viz","vol","vols","vs","w","want","wants","wasnt","way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole","who'll","whomever","whos","whose","widely","willing","wish","within","without","wont","words","world","wouldnt","www","x","yes","yet","youd","youre","z","zero","a's","ain't","allow","allows","apart","appear","appreciate","appropriate","associated","best","better","c'mon","c's","cant","changes","clearly","concerning","consequently","consider","considering","corresponding","course","currently","definitely","described","despite","entirely","exactly","example","going","greetings","hello","help","hopefully","ignored","inasmuch","indicate","indicated","indicates","inner","insofar","it'd","keep","keeps","novel","presumably","reasonably","second","secondly","sensible","serious","seriously","sure","t's","third","thorough","thoroughly","three","well","wonder"]
stopwords = set(stopwords)
stemmer = PorterStemmer()
doc_freq = dict()
term_freq = dict()
tfidf = dict()

def tokenization(content):
    content = [re.sub(r"[^a-zA-Z0-9]+", '', word) for word in content]
    content = [word.replace(',', '').replace('.', '').replace('\'', '').replace('\n', '') for word in content]
    content = [stemmer.stem(word) for word in content] # Stemming using Porter’s algorithm
    content = [word for word in content if word not in stopwords and len(word) != 0] # Stopword removal

    return content

def read_all_doc():
    for root, dirs, files in os.walk("IRTM"):
        for file in files:
            if ".txt" not in file:
                continue
            print(os.path.join(root, file))
            filename = os.path.join(root, file)
            corpus = set()
            with open(filename, "r") as f:
                contents = f.readlines()
                for content in contents:
                    text = content.lower()
                    text = text.split(" ")

                    text = tokenization(text)
                    for word in text:
                        term_freq[word] = term_freq.setdefault(word, 0) + 1
                    # print(text)
                    corpus.update(text)
                    
            for word in corpus:
                doc_freq[word] = doc_freq.setdefault(word, 0) + 1
    pass

def write_dictionary(doc_freq):
    cnt = 0
    with open("dictionary.txt", "w") as f:
        f.write("t_index,term,df\n")
        
        for k, v in doc_freq.items():
            f.write("{} {} {}\n".format(cnt, k, v))
            cnt += 1
        f.close()
    return 

def calc_tfidf(idf, dict_ind):
    for root, dirs, files in os.walk("IRTM"):
        for file in files:
            if ".txt" not in file:
                continue
            print(os.path.join(root, file))
            filename = os.path.join(root, file)
            weight = dict()
            with open(filename, "r") as f:
                contents = f.readlines()
                for content in contents:
                    text = content.lower()
                    text = text.split(" ")

                    text = tokenization(text)
                    for word in text:
                        weight[word] = weight.setdefault(word, 0) + 1
                f.close()
            for word in weight:
                weight[word] = weight[word] * idf[word]
            # print(weight)
            with open("tfidf/{}".format(file), "w") as f:
                f.write(str(len(weight))+ "\n")
                f.write("t_index tf-idf\n")
                for word in weight:
                    f.write("{} {}\n".format(dict_ind[word], weight[word]))
                f.close()
            # return
    pass

def construct_tfidf_vector(freq):

    pass

def cosine(doc1, doc2):
    with open("tfidf/{}.txt".format(doc1), "r") as f:
        content = f.readlines()
        content = [line.split(" ") for line in content]
        print(content)
        f.close()


    pass



if __name__ == "__main__":
    read_all_doc()
    
    doc_freq = {k: v for k, v in sorted(doc_freq.items(), key=lambda item: item[0])}
    write_dictionary(doc_freq)
    inv_doc_freq = {k: np.log10(1095/ v) for k, v in doc_freq.items() }
    dictionary_index = {}
    cnt_ind = 0
    for word in inv_doc_freq:
        dictionary_index[word] = cnt_ind
        cnt_ind += 1

    calc_tfidf(inv_doc_freq, dictionary_index)

    pass