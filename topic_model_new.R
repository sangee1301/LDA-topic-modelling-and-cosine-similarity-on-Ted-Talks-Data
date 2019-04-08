setwd("C:/Users/Sangee/Downloads/upwork/James Collin/Task8-Topic Modeling")

library(readr)
library(tm)
library(SnowballC)
library(jsonlite)
library(caret)
library(dplyr)
library(tidyverse)
library(topicmodels)
library(tidytext)
library(ggplot2)
library(wordcloud)


##### Data Pre-processing #####

# read data and merge
main_data <- read.csv('ted_main.csv')
transcript_data <- read.csv('transcripts.csv')
ted_talk <- merge(main_data, transcript_data, by = 'url')

### data cleaning ###
# create a corpus
ted_corpus <- VCorpus(VectorSource(ted_talk$transcript))
# standardize to lowercase characters
ted_corpus_clean <- tm_map(ted_corpus,content_transformer(tolower))
# remove stopwords
ted_corpus_clean <- tm_map(ted_corpus_clean, removeWords, stopwords("english"))
# remove punctuation
ted_corpus_clean <- tm_map(ted_corpus_clean,removePunctuation)
# stemming process to transform words to the base form
ted_corpus_clean <- tm_map(ted_corpus_clean, stemDocument)
# remove numbers
ted_corpus_clean <- tm_map(ted_corpus_clean, removeNumbers)
# remove additional whitespace
ted_corpus_clean <- tm_map(ted_corpus_clean, stripWhitespace)

# convert coorpus to DTM
ted_dtm <- DocumentTermMatrix(ted_corpus_clean)

##### Topic Modelling #####
model_lda <- LDA(ted_dtm, k = 10, method = "Gibbs",
                 control = list(verbose = 1, iter = 100))
perplexity(model_lda, ted_dtm, use_theta = TRUE, estimate_theta = TRUE)

model_lda_td <- tidy(model_lda, matrix = c("beta"))

#top 5 terms of each topic
top_terms <- model_lda_td %>%
  group_by(topic) %>%
  top_n(5, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)
top_terms

#plotting top5 terms of the 10 topics
top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_bar(alpha = 0.8, stat = "identity", show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free", ncol = 2) +
  coord_flip()

#Finding optimal no. of topics - lda tuning
library("ldatuning")
part_ted_dtm <- ted_dtm[1:500,]
result <- FindTopicsNumber(
  part_ted_dtm,
  topics = seq(from = 5, to = 30, by = 1),
  metrics = c("Griffiths2004"),
  method = "Gibbs",
  control = list(seed = 77),
  mc.cores = 2L,
  verbose = TRUE
)
FindTopicsNumber_plot(result)
#From the plot the optimal no. of topics is k=14

#lda with k=14
result <- LDA(ted_dtm, k = 14, method = "Gibbs",
                 control = list(verbose = 1, iter = 100))
result_td <- tidy(result, matrix = c("beta"))

#top 5 terms of each topic
top_terms <- result_td %>%
  group_by(topic) %>%
  top_n(5, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)
top_terms

#plotting top5 terms of the 14 topics
top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_bar(alpha = 0.8, stat = "identity", show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free", ncol = 2) +
  coord_flip()

#get probability of each topic in each doc
topicProbabilities <- as.data.frame(result@gamma)
summary(topicProbabilities)

# For each doc finding the top-ranked topic   
toptopics <- as.data.frame(cbind(document = row.names(topicProbabilities), 
                                 topic = apply(topicProbabilities,1,function(x) names(topicProbabilities)[which(x==max(x))])))
toptopics[1:100,]   

#Topic probabilities
tmResult <- posterior(result)
theta <- tmResult$topics
beta <- tmResult$terms
# re-rank top topic terms for topic names
topicNames <- apply(lda::top.topic.words(beta, 5, by.score = T), 2, paste, collapse = " ")
# most probable topics 
topicProportions <- colSums(theta) / nDocs(ted_dtm)  
names(topicProportions) <- topicNames     
sort(topicProportions, decreasing = TRUE) 

#Visualization of top words of topics 
topicToViz <- 12 # change for your own topic of interest
top40terms <- sort(tmResult$terms[topicToViz,], decreasing=TRUE)[1:40]
words <- names(top40terms)
# extract the probabilites of each of the 40 terms
probabilities <- sort(tmResult$terms[topicToViz,], decreasing=TRUE)[1:40]
# visualize the terms as wordcloud
mycolors <- brewer.pal(8, "Dark2")
wordcloud(words, probabilities, random.order = FALSE, color = mycolors)


#### Document Similarity ####
new_ted_talk <- ted_talk %>% 
  mutate(text=gsub("(http|https).+$|\\n|&amp|[[:punct:]]","",transcript),
         rowIndex=as.numeric(row.names(.))) %>% 
  select(transcript,title, main_speaker, views, url)
new_ted_talk <- tibble::rowid_to_column(new_ted_talk, "Index")

#transforming documents as list
docList <- as.list(new_ted_talk$transcript)
N.docs <- length(docList)

#Building a search function to display top 10 similar talks of the search query
QrySearch <- function(queryTerm) {
  # storing docs in Corpus class 
  my.docs <- VectorSource(c(docList, queryTerm))
  
  # Transform/standaridze docs to get ready for analysis
  my.corpus <- VCorpus(my.docs) %>% 
    tm_map(stemDocument) %>%
    tm_map(removeNumbers) %>% 
    tm_map(content_transformer(tolower)) %>% 
    tm_map(removeWords,stopwords("en")) %>%
    tm_map(stripWhitespace)
  
  # Store docs into a term document matrix where rows=terms and cols=docs
  # Normalize term counts by applying TDiDF weightings
  term.doc.matrix.stm <- TermDocumentMatrix(my.corpus,
                                            control=list(
                                              weighting=function(x) weightSMART(x,spec="ltc"),
                                              wordLengths=c(1,Inf)))
  
  # Transform term document matrix into a dataframe
  term.doc.matrix <- tidy(term.doc.matrix.stm) %>% 
    group_by(document) %>% 
    mutate(vtrLen=sqrt(sum(count^2))) %>% 
    mutate(count=count/vtrLen) %>% 
    ungroup() %>% 
    select(term:count)
  docMatrix <- term.doc.matrix %>% 
    mutate(document=as.numeric(document)) %>% 
    filter(document<N.docs+1)
  qryMatrix <- term.doc.matrix %>% 
    mutate(document=as.numeric(document)) %>% 
    filter(document>=N.docs+1)
  
  # Calcualte top ten results by cosine similarity
  searchRes <- docMatrix %>% 
    inner_join(qryMatrix,by=c("term"="term"),
               suffix=c(".doc",".query"))%>%   
    mutate(termScore=round(count.doc*count.query,4)) %>% 
    group_by(document.query,document.doc)  %>% 
    summarise(Score=sum(termScore)) %>% 
    filter(row_number(desc(Score))<=10) %>% 
    arrange(desc(Score)) %>% 
    left_join(new_ted_talk,by=c("document.doc"="Index")) %>% 
    ungroup() %>% 
    rename(Top_titles=title) %>% 
    select(Top_titles,Score) %>% 
    data.frame()
  return(searchRes)
  
}

# Give any query/topics as argument of QrySearch 
QrySearch("A young scientist's quest for clean water")#outputs top 10 similar talks
  