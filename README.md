# Quora Question Pair Similarity

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.  so main aim of project is that predicting whether pair of questions are similar or not. This could be useful to instantly provide answers to questions that have already been answered.
   Credits: Kaggle
### Problem Statement :
Identify which questions asked on Quora are duplicates of questions that have already been asked.

### Real world/Business Objectives and Constraints :
   - The cost of a mis-classification can be very high.
   - You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.
   - No strict latency concerns.
   - Interpretability is partially important.

### Performance Metric:
   - log-loss 
   - Binary Confusion Matrix

### Data Overview:
Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate. Total we have 404290 entries. Splitted data into train and test with 70% and 30%.

i derived some features from questions like no of common words, word share and some distances between questions with the help of word vectors. will discuss those below. You can check my total work [here](https://github.com/UdiBhaskar/Quora-Question-pair-similarity/blob/master/Quora%20Question%20pair%20similarity.ipynb)
### Some Analysis:
- ##### Distribution of data points among output classes  
  ![image](https://github.com/shubham-shetty12/Quora_Question_Pair/assets/137090796/b52f7b49-8d1d-49d4-8628-b1bd58fc152c)

- ##### Number of unique questions
  ![image](https://github.com/shubham-shetty12/Quora_Question_Pair/assets/137090796/2f9aeffb-6cf8-44cb-a92c-db2ac8887e31)

- ##### Number of occurrences of each question
   ![Number of occurrences of each question](https://github.com/UdiBhaskar/Quora-Question-pair-similarity/blob/master/Images/output_39_1.png "Number of occurrences of each question")
- ##### There is no duplicate pairs. Have 2 Null values, which are filled with space.
- ##### Wordcloud for similar questions
  ![image](https://github.com/shubham-shetty12/Quora_Question_Pair/assets/137090796/e82bfdd9-fa22-417e-9442-5fbcdf282c93)

- ##### Wordcloud for dissimilar questions
  ![image](https://github.com/shubham-shetty12/Quora_Question_Pair/assets/137090796/aabd9f72-a797-44ce-aad8-b6f54c860701)

### Feature Extraction:
- ##### Basic Features - Extracted some features before cleaning of data as below.
  - <b>freq_qid1</b> = Frequency of qid1's
  - <b>freq_qid2</b> = Frequency of qid2's
  - <b>q1len</b> = Length of q1
  - <b>q2len</b> = Length of q2
  - <b>q1_n_words</b> = Number of words in Question 1
  - <b>q2_n_words</b> = Number of words in Question 2
  - <b>word_Common</b> = (Number of common unique words in Question 1 and Question 2)
  - <b>word_Total</b> =(Total num of words in Question 1 + Total num of words in Question 2)
  - <b>word_share</b> = (word_common)/(word_Total)
  - <b>freq_q1+freq_q2</b> = sum total of frequency of qid1 and qid2
  - <b>freq_q1-freq_q2</b> = absolute difference of frequency of qid1 and qid2
- ##### Advanced Features - Did some preprocessing of texts and extracted some other features. i am giving some definitions which are used below. `Token`- You get a token by splitting sentence by space  ,  `Stop_Word` - stop words as per NLTK, `Word `-A token that is not a stop_word.
  - <b>cwc_min</b> = common_word_count / (min(len(q1_words), len(q2_words)) 
  - <b>cwc_max</b> = common_word_count / (max(len(q1_words), len(q2_words)) 
  - <b>csc_min</b> = common_stop_count / (min(len(q1_stops), len(q2_stops)) 
  - <b>csc_max</b> = common_stop_count / (max(len(q1_stops), len(q2_stops)) 
  - <b>ctc_min</b> = common_token_count / (min(len(q1_tokens), len(q2_tokens)) 
  - <b>ctc_max</b> = common_token_count / (max(len(q1_tokens), len(q2_tokens)) 
  - <b>last_word_eq</b> = Check if Last word of both questions is equal or not (int(q1_tokens[-1] == q2_tokens[-1]))
  - <b>first_word_eq</b> = Check if First word of both questions is equal or not (int(q1_tokens[0] == q2_tokens[0]) )
  - <b>abs_len_diff</b> = abs(len(q1_tokens) - len(q2_tokens))
  - <b>mean_len</b> = (len(q1_tokens) + len(q2_tokens))/2
  - <b>fuzz_ratio</b> = How much percentage these two strings are similar, measured with edit distance.
  - <b>fuzz_partial_ratio</b> = if two strings are of noticeably different lengths, we are getting the score of the best matching lowest length substring.
  - <b>token_sort_ratio</b> = sorting the tokens in string and then scoring fuzz_ratio.
  - <b>longest_substr_ratio</b> = len(longest common substring) / (min(len(q1_tokens), len(q2_tokens))
- ##### Extracted Tf-Idf features for this combained question1 and question2 and got 1,2,3 gram features with Train data. Transformed test data into same vector space. 
- ##### Got [Word Movers Distance](http://proceedings.mlr.press/v37/kusnerb15.pdf) with pretrained glove word vectors. 
- ##### From Pretrained glove word vectors got average word vector for question1 and question2. With this avg word vector got below distances. 
  - <b>Cosine distance</b>
  - <b>Cityblock distance</b>
  - <b>Canberra distance</b>
  - <b>Euclidean distance</b>
  - <b>Minkowski distance</b>
### Some Features analysis and visualizations:
- ##### word_share - We can check from below that it is overlaping a bit, but it is giving some classifiable score for disimilar questions.
   ![image](https://github.com/shubham-shetty12/Quora_Question_Pair/assets/137090796/22f3eef3-d8e2-4ddb-aa08-fe6b40c5f510)

- ##### Word Common - it is almost overlaping.
  ![image](https://github.com/shubham-shetty12/Quora_Question_Pair/assets/137090796/a0b43c11-159a-4275-9842-15910896d1c5)

- ##### Bivariate analysis of features 'ctc_min', 'cwc_min', 'csc_min', 'token_sort_ratio'. We can observe that we can divide duplicate and non duplicate with some of these features with some patterns. 
 ![image](https://github.com/shubham-shetty12/Quora_Question_Pair/assets/137090796/c4387b7c-c5db-4b12-9d26-61f853075914)

### Machine Learning Models:
   - Trained a random model to check Worst case log loss and got log loss as 0.8957
   - Trained some models and also tuned hyperparameters using Random search. I didnt use total train data to train my algorithms, because of ram constraint in my PC, i sampled 25K datapoints for training my models. below are models and their logloss scores.
   For below table BF - Basic features, AF - Advanced features, DF - Distance Features including WMD.

| Model               |  Log Loss |
| -------------       |  ------------- |
| Logistic Regression | 0.55685 |
| Linear SVM          | 0.6601  |
| XGBoost             | 0.4343  |

## Conclusion:
Multiple algorithms were applied to the preprocessed data, and XGBoost emerged as the top-performing model with a log loss of 0.4343. The successful implementation of XGBoost demonstrated its effectiveness in handling this specific problem. Overall, the project aimed to tackle the challenge of identifying question pairs with similar meanings, and through careful feature engineering and model selection, a reliable and accurate solution was achieved.

##### References:
1. https://www.kaggle.com/c/quora-question-pairs 
2. Applied AI Course: https://www.appliedaicourse.com/
3. https://github.com/seatgeek/fuzzywuzzy#usage 
4. https://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
