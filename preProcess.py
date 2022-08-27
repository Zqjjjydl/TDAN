import torch
from gensim import corpora
import nltk
from nltk.corpus import stopwords
from gensim.models import LdaMulticore
from gensim import corpora
from gensim.models import KeyedVectors
import numpy as np
domain_class=["books","dvd","electronics","kitchen","yelp"]

max_len_bow=176
k=50
threhold_specific=0.08
specific_token='<specific_token>'

def get_bow_for_domain(specific_topic,bow,dictionary,dictionary_idx):
    specific_word_doc=[]
    topic_vec=[]
    paddingIdx=dictionary_idx.token2id["<padding>"]
    for doc in bow:
        topic_dis_vec=torch.zeros(k,dtype=torch.float)
        topic_dis,topic_per_word,phi_per_word=lda.get_document_topics(doc,per_word_topics=True)
        for topic in topic_dis:
            topic_dis_vec[topic[0]]=float(topic[1])
        specific_word=[]#specific word
        for topic_and_word in topic_per_word:#for each word
            if len(topic_and_word[1])==0:
                continue
            word=dictionary_idx.token2id[dictionary[topic_and_word[0]]]
            if topic_and_word[1][0] in specific_topic:
                specific_word.append(word)
        specific_word.append(dictionary_idx.token2id[specific_token])
        specific_word_doc.append(specific_word)
        topic_vec.append(topic_dis_vec)
    #add padding
    for i in range(len(specific_word_doc)):
        if len(specific_word_doc[i])<max_len_bow:
            tempLen=max_len_bow-len(specific_word_doc[i])
            tempArr=tempLen*[paddingIdx]
            specific_word_doc[i].extend(tempArr)
        else:
            specific_word_doc[i]=specific_word_doc[i][:max_len_bow]
    return specific_word_doc,topic_vec
def generateInput(source_domain,target_domain,lda,bow,dictionary,dictionary_idx):
    source_bow=bow[source_domain]
    target_bow=bow[target_domain]
    #calculate average topic occurance for each domain
    source_topic_dis=np.zeros(k,dtype=float)
    for doc in source_bow:
        topic_dis=lda.get_document_topics(doc)
        for item in topic_dis:
            source_topic_dis[item[0]]+=item[1]
    source_topic_dis/=len(source_bow)

    target_topic_dis=np.zeros(k,dtype=float)
    for doc in target_bow:
        topic_dis=lda.get_document_topics(doc)
        for item in topic_dis:
            target_topic_dis[item[0]]+=item[1]
    target_topic_dis/=len(source_bow)
    topic_diff=source_topic_dis-target_topic_dis
    specific_topic_source=set()
    specific_topic_target=set()
    for topic in range(k):
        if topic_diff[topic]>threhold_specific:
            specific_topic_source.add(topic)
        elif topic_diff[topic]<-threhold_specific:
            specific_topic_target.add(topic)
    specific_bow_source,topic_vec_source=get_bow_for_domain(specific_topic_source,source_bow,dictionary,dictionary_idx)
    specific_bow_target,topic_vec_target=get_bow_for_domain(specific_topic_target,target_bow,dictionary,dictionary_idx)
    torch.save({"specific_bow_source":specific_bow_source,
                "topic_vec_source":topic_vec_source,
                "topic_vec_target":topic_vec_target,
                "specific_bow_target":specific_bow_target
                },f="./processedData/TAN_input_"+domain_class[source_domain]+domain_class[target_domain])

    torch.save({"specific_bow_source":specific_bow_target,
                "topic_vec_source":topic_vec_target,
                "specific_bow_target":specific_bow_source,
                "topic_vec_target":topic_vec_source
                },f="./processedData/TAN_input_"+domain_class[target_domain]+domain_class[source_domain])

if __name__ == '__main__':
    
    labeled_text=[]
    label=[]

    for domain in domain_class:
        domain_text=[]
        domain_label=[]
        with open("./raw_data/"+domain+"/review_negative","r",encoding="utf-8") as f:
            tempArr=f.readlines()
            domain_text=domain_text+tempArr
            domain_label=domain_label+[0 for i in range(len(tempArr))]
        with open("./raw_data/"+domain+"/review_positive","r",encoding="utf-8") as f:
            tempArr=f.readlines()
            domain_text=domain_text+tempArr
            domain_label=domain_label+[1 for i in range(len(tempArr))]
        labeled_text.append(domain_text)
        label.append(domain_label)

    torch.save({
        'label':label
    },'./processedData/label')

    temp_labeled_text=[]
    print("----------word tokenize---------")
    for i in range(len(domain_class)):
        print(i)
        temp_labeled_text.append([nltk.word_tokenize(document.lower().strip()) for document in labeled_text[i]])  
    labeled_text=temp_labeled_text

    #padding
    print("----------padding building---------")
    #trunc
    maxLen=500

    for i in range(len(domain_class)):
        print(i)
        for j in range(len(labeled_text[i])):#for each document
            tempLen=maxLen-len(labeled_text[i][j])
            if tempLen<0:
                labeled_text[i][j]=labeled_text[i][j][0:maxLen]
            elif tempLen>0:
                tempArr=tempLen*["<padding>"]
                labeled_text[i][j].extend(tempArr)

    texts=[]
    for i in range(len(domain_class)):
        texts=texts+labeled_text[i]

    dictionary_idx = corpora.Dictionary(texts)
    dictionary_idx.add_documents([["<specific_token>"]])

    labeled_text_inIdx=[]
    print("----------doc 2 idx---------")
    for i in range(len(domain_class)):
        print(i)
        labeled_text_inIdx.append([dictionary_idx.doc2idx(document) for document in labeled_text[i]]) 
    
    print(dictionary_idx.token2id)#!
    print(dictionary_idx.id2token)#!
    dictionary_idx.save('./processedData/idxSentenceDict.dict')
    torch.save({
        'labeled_text_inIdx':labeled_text_inIdx,
        'unlabeled_text_inIdx':[]
    },'./processedData/dataInIdx')


    mystopwords=[]
    wv = KeyedVectors.load_word2vec_format("C:/Users/74158/Desktop/paper/code/Gated DERNN-GRU-Topic/word2vec/GoogleNews-vectors-negative300.bin", binary=True)

    word_num=len(dictionary_idx)
    wordvec_matrix=torch.zeros((word_num,300))
    print("----------wordvec building",len(dictionary_idx),"to building","---------")
    for idx,word in dictionary_idx.items():
        if idx%10000==0:
            print(idx)
        if word in wv:
            wordvec_matrix[idx]=torch.tensor(wv[word])
        else:
            mystopwords.append(word)
            wordvec_matrix[idx]=(10**-2)*torch.randn(300)

    torch.save({
        'idx2vec':wordvec_matrix,
        'idx2word':dictionary_idx.id2token,
        'token2id':dictionary_idx.token2id
    },'./processedData/dictionary')

    dictionary = corpora.Dictionary(texts)
    mystopwords = stopwords.words('english')
    for w in ['!',',','.','?','-s','-ly','</s>'\
        ,'s','(',')',"*","\'s","would"\
            ,"could","``","\'\'","n\'t","one",";","&",":","-"\
                ,"also","\'","--","\'ve","\'m"\
                    ,"-","\'re","\'ll","\'d",\
                        "\'the","\'how","\'what","\'must","\'it",\
                            "\'new","us","$","mr.","[","]","`","#","oh",\
                            "though"]:
        mystopwords.append(w)
    for i in range(10):
        mystopwords.append('.'*i)
    mystopwords.append("<padding>")
    print("stopword count:",len(mystopwords))
    print("stop words: ",mystopwords)
    badid=[idx for idx in dictionary if dictionary[idx].lower() in mystopwords]
    dictionary.filter_tokens(bad_ids=badid)
    dictionary.filter_extremes(no_below=5,no_above=1)

    #build bow
    print("----------bow building",len(dictionary),"words to building","---------")
    labeled_text_bow = []
    bow_to_save=[]
    for i in range(len(domain_class)):
        print(i)
        temp=[dictionary.doc2bow(document) for document in labeled_text[i]]
        bow_to_save.append(temp)
        labeled_text_bow+=temp
    torch.save(bow_to_save,f="./processedData/labeled_bow")
    torch.save(dictionary,f="./processedData/labeled_bow_dictionnary")
    print(dictionary[0])

    lda=LdaMulticore(labeled_text_bow,workers=4,num_topics=50,id2word=dictionary,passes=200)
    # lda=LdaMulticore.load("./model/ldaModel")

    topics=lda.print_topics(num_words=10)
    print(topics)
    lda.save("./model/ldaModel")

    for source_domain in range(4):
        for target_domain in range(source_domain+1,5):
            print("source: ",source_domain,"target: ",target_domain)
            generateInput(source_domain,target_domain,lda,bow_to_save,dictionary,dictionary_idx)
    