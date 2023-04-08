
# 경고 무시
import warnings
warnings.filterwarnings('ignore')

# 패키지, 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import nltk
from konlpy.tag import Okt
import re
import tensorflow as tf
from tensorflow import keras
from keybert import KeyBERT
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from transformers import TFBertModel, PreTrainedTokenizer
import transformers
import tensorflow_addons as tfa
import torch
from sentence_transformers import SentenceTransformer, util
from torch import tensor
from wordcloud import WordCloud
from PIL import Image
from krwordrank.hangle import normalize
from krwordrank.word import KRWordRank

import logging
import os
import unicodedata
from shutil import copyfile

from transformers import PreTrainedTokenizer


def text_rec_run(spot, kakao, method="kakao"):    

    print('tensorflow-gpu:',tf.test.is_gpu_available())
    print('torch-gpu:',torch.cuda.is_available())

    USE_CUDA = torch.cuda.is_available()
    device = torch.device('cuda:0' if USE_CUDA else 'cpu')
    print(f'device: {device}')

    # 장소 데이터
    place_data = 'chat/static/csv/review_text_csv/review_result_50_nouns(2).csv'

    # 채팅 감정 분류기
    sentiment_model = 'chat/static/model/model.h5'

    # 불용어 data
    stopwords_path = 'chat/static/csv/stopwords_final.csv'

    # 장소별 키워드 사전
    keyword_dict = 'chat/static/csv/키워드 사전_final(2).csv'

    # 리뷰 data 임베딩 사전학습 모델 
    sts_bert_model = "jhgan/ko-sroberta-multitask"

    # 워드클라우드 mask 이미지 경로
    mask_path = 'chat/static/img/wc_mask.png'

    # 해당 장소 데이터 불러오기
    data = pd.read_csv(place_data, encoding='utf-8-sig')
    df = data[data['address'].str.contains(spot)].reset_index(drop=True)


    # 채팅 감성분석 모델
    # coding=utf-8
    # Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team and Jangwon Park
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    """ Tokenization classes for KoBert model."""



    logger = logging.getLogger(__name__)

    VOCAB_FILES_NAMES = {"vocab_file": "tokenizer_78b3253a26.model",
                        "vocab_txt": "vocab.txt"}

    PRETRAINED_VOCAB_FILES_MAP = {
        "vocab_file": {
            "monologg/kobert": "https://s3.amazonaws.com/models.huggingface.co/bert/monologg/kobert/tokenizer_78b3253a26.model",
            "monologg/kobert-lm": "https://s3.amazonaws.com/models.huggingface.co/bert/monologg/kobert-lm/tokenizer_78b3253a26.model",
            "monologg/distilkobert": "https://s3.amazonaws.com/models.huggingface.co/bert/monologg/distilkobert/tokenizer_78b3253a26.model"
        },
        "vocab_txt": {
            "monologg/kobert": "https://s3.amazonaws.com/models.huggingface.co/bert/monologg/kobert/vocab.txt",
            "monologg/kobert-lm": "https://s3.amazonaws.com/models.huggingface.co/bert/monologg/kobert-lm/vocab.txt",
            "monologg/distilkobert": "https://s3.amazonaws.com/models.huggingface.co/bert/monologg/distilkobert/vocab.txt"
        }
    }

    PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
        "monologg/kobert": 512,
        "monologg/kobert-lm": 512,
        "monologg/distilkobert": 512
    }

    PRETRAINED_INIT_CONFIGURATION = {
        "monologg/kobert": {"do_lower_case": False},
        "monologg/kobert-lm": {"do_lower_case": False},
        "monologg/distilkobert": {"do_lower_case": False}
    }

    SPIECE_UNDERLINE = u'▁'


    class KoBertTokenizer(PreTrainedTokenizer):
        """
            SentencePiece based tokenizer. Peculiarities:
                - requires `SentencePiece <https://github.com/google/sentencepiece>`_
        """
        vocab_files_names = VOCAB_FILES_NAMES
        pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
        pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
        max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

        def __init__(
                self,
                vocab_file,
                vocab_txt,
                do_lower_case=False,
                remove_space=True,
                keep_accents=False,
                unk_token="[UNK]",
                sep_token="[SEP]",
                pad_token="[PAD]",
                cls_token="[CLS]",
                mask_token="[MASK]",
                **kwargs):
            super().__init__(
                unk_token=unk_token,
                sep_token=sep_token,
                pad_token=pad_token,
                cls_token=cls_token,
                mask_token=mask_token,
                **kwargs
            )

            # Build vocab
            self.token2idx = dict()
            self.idx2token = []
            with open(vocab_txt, 'r', encoding='utf-8') as f:
                for idx, token in enumerate(f):
                    token = token.strip()
                    self.token2idx[token] = idx
                    self.idx2token.append(token)

            try:
                import sentencepiece as spm
            except ImportError:
                logger.warning("You need to install SentencePiece to use KoBertTokenizer: https://github.com/google/sentencepiece"
                            "pip install sentencepiece")

            self.do_lower_case = do_lower_case
            self.remove_space = remove_space
            self.keep_accents = keep_accents
            self.vocab_file = vocab_file
            self.vocab_txt = vocab_txt

            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.Load(vocab_file)

        @property
        def vocab_size(self):
            return len(self.idx2token)

        def get_vocab(self):
            return dict(self.token2idx, **self.added_tokens_encoder)

        def __getstate__(self):
            state = self.__dict__.copy()
            state["sp_model"] = None
            return state

        def __setstate__(self, d):
            self.__dict__ = d
            try:
                import sentencepiece as spm
            except ImportError:
                logger.warning("You need to install SentencePiece to use KoBertTokenizer: https://github.com/google/sentencepiece"
                            "pip install sentencepiece")
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.Load(self.vocab_file)

        def preprocess_text(self, inputs):
            if self.remove_space:
                outputs = " ".join(inputs.strip().split())
            else:
                outputs = inputs
            outputs = outputs.replace("``", '"').replace("''", '"')

            if not self.keep_accents:
                outputs = unicodedata.normalize('NFKD', outputs)
                outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
            if self.do_lower_case:
                outputs = outputs.lower()

            return outputs

        def _tokenize(self, text, return_unicode=True, sample=False):
            """ Tokenize a string. """
            text = self.preprocess_text(text)

            if not sample:
                pieces = self.sp_model.EncodeAsPieces(text)
            else:
                pieces = self.sp_model.SampleEncodeAsPieces(text, 64, 0.1)
            new_pieces = []
            for piece in pieces:
                if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                    cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(SPIECE_UNDERLINE, ""))
                    if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                        if len(cur_pieces[0]) == 1:
                            cur_pieces = cur_pieces[1:]
                        else:
                            cur_pieces[0] = cur_pieces[0][1:]
                    cur_pieces.append(piece[-1])
                    new_pieces.extend(cur_pieces)
                else:
                    new_pieces.append(piece)

            return new_pieces

        def _convert_token_to_id(self, token):
            """ Converts a token (str/unicode) in an id using the vocab. """
            return self.token2idx.get(token, self.token2idx[self.unk_token])

        def _convert_id_to_token(self, index, return_unicode=True):
            """Converts an index (integer) in a token (string/unicode) using the vocab."""
            return self.idx2token[index]

        def convert_tokens_to_string(self, tokens):
            """Converts a sequence of tokens (strings for sub-words) in a single string."""
            out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
            return out_string

        def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
            """
            Build model inputs from a sequence or a pair of sequence for sequence classification tasks
            by concatenating and adding special tokens.
            A KoBERT sequence has the following format:
                single sequence: [CLS] X [SEP]
                pair of sequences: [CLS] A [SEP] B [SEP]
            """
            if token_ids_1 is None:
                return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
            cls = [self.cls_token_id]
            sep = [self.sep_token_id]
            return cls + token_ids_0 + sep + token_ids_1 + sep

        def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
            """
            Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
            special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.
            Args:
                token_ids_0: list of ids (must not contain special tokens)
                token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                    for sequence pairs
                already_has_special_tokens: (default False) Set to True if the token list is already formated with
                    special tokens for the model
            Returns:
                A list of integers in the range [0, 1]: 0 for a special token, 1 for a sequence token.
            """

            if already_has_special_tokens:
                if token_ids_1 is not None:
                    raise ValueError(
                        "You should not supply a second sequence if the provided sequence of "
                        "ids is already formated with special tokens for the model."
                    )
                return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

            if token_ids_1 is not None:
                return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
            return [1] + ([0] * len(token_ids_0)) + [1]

        def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
            """
            Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
            A KoBERT sequence pair mask has the following format:
            0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence
            if token_ids_1 is None, only returns the first portion of the mask (0's).
            """
            sep = [self.sep_token_id]
            cls = [self.cls_token_id]
            if token_ids_1 is None:
                return len(cls + token_ids_0 + sep) * [0]
            return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

        def save_vocabulary(self, save_directory):
            """ Save the sentencepiece vocabulary (copy original file) and special tokens file
                to a directory.
            """
            if not os.path.isdir(save_directory):
                logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
                return

            # 1. Save sentencepiece model
            out_vocab_model = os.path.join(save_directory, VOCAB_FILES_NAMES["vocab_file"])

            if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_model):
                copyfile(self.vocab_file, out_vocab_model)

            # 2. Save vocab.txt
            index = 0
            out_vocab_txt = os.path.join(save_directory, VOCAB_FILES_NAMES["vocab_txt"])
            with open(out_vocab_txt, "w", encoding="utf-8") as writer:
                for token, token_index in sorted(self.token2idx.items(), key=lambda kv: kv[1]):
                    if index != token_index:
                        logger.warning(
                            "Saving vocabulary to {}: vocabulary indices are not consecutive."
                            " Please check that the vocabulary is not corrupted!".format(out_vocab_txt)
                        )
                        index = token_index
                    writer.write(token + "\n")
                    index += 1

            return out_vocab_model, out_vocab_txt
        
    # try, except로 예외처리, for문 이용하여 대화내용 list에 담기
    content_all=[]
    if method == "kakao":
        for i in kakao[kakao.columns[0]]:
            content=i.split(',', 2)
            try:
                content2=content[1].split(':', 2)
                content_all.append(content[0] + ',' + content2[0] + ',' + content2[1])
            except:
                continue

        get_date=[]
        get_content=[]
        for i in content_all:
            ii=i.split(',', 3)
            get_date.append(ii[0])
            get_content.append(ii[2])

        kakao_df=pd.DataFrame(data=get_date, columns=['date'])
        kakao_df['content']=get_content

        chat = ''.join(get_content)
    else:
        chat = ' '.join(kakao)
        kakao_df=pd.DataFrame(data=kakao, columns=['content'])

    # 메시지 전처리 함수
    def message_cleaning(docs):

        # Series의 object를 str로 변경
        docs=[str(doc) for doc in docs]

        # 1. 쓸모없는 단어 삭제
        pattern1=re.compile("<사진|동영상")
        docs=[pattern1.sub("", doc) for doc in docs] 

        pattern2=re.compile("읽지|않음>")
        docs=[pattern2.sub("", doc) for doc in docs] 

        pattern3=re.compile(r"[\w]+.jpg")
        docs=[pattern3.sub("", doc) for doc in docs]

        # 2. 단순 자음, 모음 삭제
        #pattern4=re.compile("[ㄱ-ㅎ]*[ㅏ-ㅢ]*")
        #docs=[pattern4.sub("", doc) for doc in docs] 

        # 3. 링크로 되어있는 글 삭제
        pattern5=re.compile(r"\b(https?:\/\/)?([\w.]+){1, 2}(\.[\w]{2, 4}){1, 2}(.*)")
        docs=[pattern5.sub("", doc) for doc in docs] 

        # 4. 특수문자 삭제
        # pattern6=re.compile("[\{\}\[\]\/?.,;:|\)*~`!^\-_+<>@\#$%&\\\=\(\'\"]")
        # docs=[pattern6.sub("", doc) for doc in docs] 

        # 5. 이모티콘 삭제
        pattern7=re.compile("이모티콘")
        docs=[pattern7.sub("", doc) for doc in docs]

        # 6. ㅋ 삭제
        pattern8=re.compile("ㅋ")
        docs=[pattern8.sub("", doc) for doc in docs]


        return docs

    # 카카오톡 텍스트 정제
    content_series=kakao_df['content']
    cleaned_corpus=message_cleaning(content_series)

    # 띄어쓰기 된 부분 앞뒤로 붙여주기
    cleaned_corpus2=[]
    for i in cleaned_corpus:
        cleaned_corpus2.append(i.strip())

    # 단어 삭제 후 빈 행 삭제
    cleaned_text=pd.Series(cleaned_corpus2)
    kakao_df['content']=cleaned_text

    kakao_df=kakao_df[kakao_df['content']!=""]

    # 중복 제거
    kakao_df.drop_duplicates(subset = ['content'], inplace=True)

    kakao_df=kakao_df.reset_index(drop=True)
    
    
    # 모델 로드
    model = keras.models.load_model(sentiment_model,
                                    custom_objects={"TFBertModel": transformers.TFBertModel})


    kakao_df['label']=2

    DATA_COLUMN='content'
    LABEL_COLUMN='label'

    SEQ_LEN = 64 # SEQ_LEN: input length

    def load_data(pandas_dataframe):
        data_df = pandas_dataframe
        data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
        data_df[LABEL_COLUMN] = data_df[LABEL_COLUMN].astype(int)
        data_x, data_y = convert_data(data_df)
        return data_x, data_y
        
    # 네이버 영화리뷰 문장 Bert input 형식을 변환
    def convert_data(data_df):
        # global tokenizer
        tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
        tokens, masks, segments, targets=[], [], [], []

        for i in tqdm(range(len(data_df))):
            # token: Tokenize the sentence
            token=tokenizer.encode(data_df[DATA_COLUMN][i], truncation=True, pad_to_max_length=True, max_length=SEQ_LEN)

            # 마스크는 토큰화한 문장에서 패딩이 아닌 부분은 1, 패딩인 부분은 0으로 통일
            num_zeros = token.count(0)
            mask = [1]*(SEQ_LEN-num_zeros) + [0]*num_zeros
            
            # 문장의 전후관계를 구분해주는 세그먼트는 문장이 1개밖에 없으므로 모두 0
            segment = [0]*SEQ_LEN
    
            # 버트 인풋으로 들어가는 token, mask, segment를 tokens, segments에 각각 저장
            tokens.append(token)
            masks.append(mask)
            segments.append(segment)
            
            # 정답(긍정 : 1 부정 0)을 targets 변수에 저장해 줌
            targets.append(data_df[LABEL_COLUMN][i])
    
        # tokens, masks, segments, 정답 변수 targets를 numpy array로 지정    
        tokens = np.array(tokens)
        masks = np.array(masks)
        segments = np.array(segments)
        targets = np.array(targets)
    
        return [tokens, masks, segments], targets

    # tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')

    test_x2,text_y2=load_data(kakao_df)

    pred=model.predict(test_x2)
    kakao_df['result']=pred

    kakao_df['label']='-'

    # 0: 중립(제안)
    suggestion=['갈까', '갈까요', '볼까', '볼까요', '먹을까', '먹을까요', '할까', '할까요', '갈래', '갈래요',  '볼래', '볼래요', '할래', 
                '할래요', '어때', '어때요', '어떠니', '먹자', '먹어요', '가자', '가요', '하자', '해요', '보자', '봐요', '?']
    for i in range(len(kakao_df)):
        for j in suggestion:
            if j in kakao_df['content'][i]:
                kakao_df['label'][i]=0
                break

    # -1: 부정, 0: 중립(제안), 1: 긍정
    for i in range(len(kakao_df)):
        if kakao_df['label'][i]!=0:             
            if kakao_df['result'][i]<0.04:
                kakao_df['label'][i]=-1
            else:
                kakao_df['label'][i]=1
        else:
            continue

    kakao_df['label']=kakao_df['label'].astype(int)

    # 불용어
    f = open(stopwords_path, 'r', encoding='utf-8')
    stopwords = f.read().split()

    # 형태소 분석기 사용하여 토큰화 하면서 불용어 제거
    okt=Okt()

    nouns=[]
    for sentence in tqdm(kakao_df['content']):
        tokenized_sentence = okt.nouns(str(sentence)) # 토큰화
        stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
        nouns.append(stopwords_removed_sentence)


    kakao_df['nouns']=nouns

    # 중립(제안) 문장 명사 추출
    neutral_word_list=[]
    for i in range(len(kakao_df)):
        if kakao_df['label'][i]==0:
            neutral_word_list.extend(kakao_df['nouns'][i])

    # 부정 문장 명사 추출
    negative_word_list=[]
    for i in range(len(kakao_df)):
        if kakao_df['label'][i]==-1:
            negative_word_list.extend(kakao_df['nouns'][i])

    # 긍정 문장 명사 추출 
    positive_word_list=[]
    for i in range(len(kakao_df)):
        if kakao_df['label'][i]==1:
            positive_word_list.extend(kakao_df['nouns'][i])

    # 중립 문장 명사 중 부정 문장 명사 삭제
    for w in negative_word_list:
        if w in neutral_word_list:
            neutral_word_list.remove(w)

    # 긍정 문장 명사 중 부정 문장 명사 삭제
    for w in negative_word_list:
        if w in positive_word_list:
            positive_word_list.remove(w)

    # word_list의 명사들의 개수
    from collections import Counter
    count=Counter(neutral_word_list)
    neutral_word_cnt=dict(count)

    from collections import Counter
    good_count=Counter(positive_word_list)
    good_word_cnt=dict(good_count)

    #중복제거
    neutral_word_list=list(set(neutral_word_list))
    positive_word_list=list(set(positive_word_list))

    # 중립 문장 
    # 키워드 사전 count 세기
    mydict=pd.read_csv(keyword_dict, encoding='utf-8')
    mydict_dict=mydict.to_dict()

    for get_words in neutral_word_list:
        for i in range(len(mydict_dict['count'])):
            cnt=0
            for j in range(len(mydict_dict['category'][i])):
                if get_words in mydict_dict['category'][i] or get_words in mydict_dict['words'][i]: 
                    if len(get_words) == 1 and get_words != mydict_dict['category'][i]:
                        continue 
                    elif '카페' in get_words and get_words != mydict_dict['category'][i]:
                        continue
                    elif '음식' in get_words and get_words != mydict_dict['category'][i]:
                        continue
                    else: 
                        cnt=neutral_word_cnt[get_words]
            mydict_dict['count'][i]+=cnt

    result_df = pd.DataFrame(mydict_dict)

    # count가 0이 아닌 카테고리만 추출
    # result_df.loc[result_df['count']!=0]

    # 긍정 문장 
    mydict2=pd.read_csv(keyword_dict, encoding='utf-8')
    mydict_dict2=mydict2.to_dict()

    for get_words in positive_word_list:
        for i in range(len(mydict_dict2['count'])):
            cnt=0
            for j in range(len(mydict_dict2['category'][i])):
                if get_words in mydict_dict2['category'][i] or get_words in mydict_dict2['words'][i]: 
                    if len(get_words) == 1 and get_words != mydict_dict2['category'][i]:
                        continue 
                    elif '카페' in get_words and get_words != mydict_dict2['category'][i]:
                        continue
                    elif '음식' in get_words and get_words != mydict_dict2['category'][i]:
                        continue
                    else:
                        cnt = good_word_cnt[get_words]
            mydict_dict2['count'][i]+=cnt

    good_result_df = pd.DataFrame(mydict_dict2)
    # good_result_df.loc[good_result_df['count']!=0]

    result_df['count'] = good_result_df['count'] + result_df['count']
    # result_df.loc[result_df['count']!=0]

    model = SentenceTransformer(sts_bert_model)

    # 코사인 유사도계산 함수
    # input: 채팅, df, 컬럼명
    def cal_cossim(chat, df, col_name):
        result_list = []
        
        # 채팅 임베딩
        chat_embedding = model.encode(str(chat))
        a = torch.tensor(chat_embedding).to(device)
        
        for i in tqdm(range(len(df))):
            # sent = str(df[col_name][i])
            sent = str(df[col_name][i])[:1500] # 분석 글자 수 제한
            
            # 비교군 임베딩
            review_embeddings = model.encode(sent)
            b = torch.tensor(review_embeddings).to(device)
            
            sim = util.pytorch_cos_sim(a, b).to(device)
            sim = sim.item()
            result_list.append(sim)
        return result_list

    # 유사도 비교
    df['name_sim'] = cal_cossim(chat, df, 'name')
    df['category_sim'] = cal_cossim(chat, df, 'category')
    df['review_sim'] = cal_cossim(chat, df, 'review')
    # df['review_sim'] = cal_cossim(chat, df, 'review_nouns')

    # 키워드 사전에서 count가 0이 아닌 카테고리만 추출
    kw_data = result_df.loc[result_df['count']!=0].reset_index(drop=True)

    # 키워드 사전 vs 카테고리
    df['chat_cnt'] = 0
    for i in range(len(kw_data)):
        words = kw_data['words'][i]
        for word in words:
            for j in range(len(df)):
                if word in df['category'][j]:
                    df['chat_cnt'][j] += kw_data['count'][i]
                    
    # 수식에 들어가는 변수 정규화 (최소:0 ~ 최대:1)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    # 수식에 들어가는 컬럼들 정규화
    cols = ['name_sim', 'category_sim', 'review_sim', 'rate', 'chat_cnt']
    x = df[cols].values
    x_scaled = scaler.fit_transform(x)
    df_temp = pd.DataFrame(x_scaled, columns=cols, index = df.index)
    df[cols] = df_temp

    # 가중치 산정
    # 별점, 가게명, 카테고리명, 리뷰, 장소사전
    w = [
        0.30, # 별점
        2.00, # 가게명
        1.20, # 카테고리명
        2.30, # 리뷰
        1.80, # 장소사전
        ]

    # 수식 결정
    df['score'] = (df['rate'] * w[0]) + (df['name_sim'] * w[1]) + (df['category_sim'] * w[2]) + (df['review_sim'] * w[3]) + (df['chat_cnt'] * w[4])
    result = df.sort_values(by='score', ascending=False).reset_index(drop=True) # score 내림차순 정렬

    # 전체 추천 결과
    result_all = result.copy()

    # 카테고리 분류
    restaurant = ['음식점>한식>육류,고기요리>닭볶음탕','양식>독일음식', '음식점>분식>불타는여고24시떡볶이',  '음식점>일식>초밥,롤', '한식>일공공키친', '감자탕>신사골옛날감자탕', '음식점>양식>핫도그', '일식>일식튀김,꼬치', '음식점>후렌치후라이', '한식>비빔밥',  '치킨,닭강정', '음식점>한식>해물,생선요리>바닷가재요리', '음식점>한식>해물,생선요리>생선회', '음식점>푸드트럭', '음식점>양식>브런치', '태국음식>바나나테이블', '음식점>뷔페>고기뷔페', '이탈리아음식>스파게티,파스타전문>스파게티스토리', '해물,생선요리>주꾸미요리>신복관', '한식>육류,고기요리>순대,순댓국', '해물,생선요리>생선회', '곱창,막창,양>안경할머니곱창', '중식>양꼬치>미친양꼬치', '육류,고기요리>오리요리', '음식점>한식>육류,고기요리>오리요리', '베트남음식>72420',  '음식점',  '양식>스테이크,립', '해물,생선요리>아귀찜,해물찜', '육류,고기요리>순대,순댓국>개성진찹쌀순대', '음식점>한식>육류,고기요리>순대,순댓국', '음식점>한식>냉면', '음식점>한식>해물,생선요리>게요리', '음식점>한식>곰탕,설렁탕', '음식점>한식>육류,고기요리>닭요리', '음식점>한식>육류,고기요리>찜닭', '음식점>일식>카레', '음식점>터키음식', '음식점>킹크랩요리', '한식>쌈밥>무영쌈밥정식', '한식>해물,생선요리>게요리>대게나라', '음식점>육류,고기요리', '일식>일본식라면>잇또라멘', '해물,생선요리>주꾸미요리', '애플김밥',  '음식점>양식>피자>서오릉피자', '분식>라면', '한식>전복요리', '음식점>한식>찌개,전골', '음식점>한식>육류,고기요리>양갈비', '한식>국밥>24시전주콩나물국밥', '육류,고기요리>닭갈비', '한식>육류,고기요리>닭발', '한식>해물,생선요리>주꾸미요리', '음식점>한식>국수', '양식>이탈리아음식', '일식>덮밥', '음식점>한식>추어탕', '한식>전,빈대떡', '음식점>양식>멕시코,남미음식', '해물,생선요리>오징어요리', '음식점>일식>우동,소바', '음식점>일식>덮밥', '해물,생선요리>바닷가재요리', '한식>족황상제', '음식점>한식>보리밥', '음식점>한식>감자탕', '한식>갈비탕', '일식>돈가스>101번지남산돈까스', '한식>사철,영양탕', '뷔페>채식,샐러드뷔페', '음식점>일식>샤브샤브', '음식점>한식>육류,고기요리>닭볶음탕' '음식점>프랑스음식', '양식>피자', '음식점>한식>육류,고기요리>곱창,막창,양', '음식점>돈가스', '음식점>한식>육류,고기요리>돼지고기구이', '음식점>한식>육류,고기요리', '한식>육류,고기요리>고기원칙', '음식점>중식>중식당', '한식>해장국>장수본가해장국', '음식점>샌드위치', '분식>호떡', '음식점>핫도그', '음식점>양식>피자', '음식점>한식>육류,고기요리>닭갈비', '음식점>스페인음식', '한식>향토음식', '양식>브런치', '양식>샌드위치', '음식점>한식>해물,생선요리>개성집', '퓨전음식', '음식점>중식>양꼬치', '음식점>일식>일본식라면', '음식점>일식>일식당', '분식>떡볶이', '음식점>한식>해물,생선요리', '음식점>한식>육류,고기요리>백숙,삼계탕', '한식>육류,고기요리>닭발>미녀닭발', '한식>닭볶음탕', '한식>장어,먹장어요리', '육류,고기요리>소고기구이', '음식점>양식>햄버거', '양식>햄버거', '한식>게요리', '해물,생선요리>대게요리', '육류,고기요리>순대,순댓국', '육류,고기요리>곱창,막창,양', '한식>생선구이', '일식>카레', '분식>토스트', '음식점>양식>이탈리아음식>스파게티,파스타전문', '음식점>한식>해물,생선요리>조개요리', '음식점>양식>이탈리아음식', '음식점>한식>해물,생선요리>주꾸미요리', '음식점>죽', '한식>죽', '음식점>스테이크,립', '음식점>멕시코,남미음식', '한식>보리밥', '한식>두부요리', '한식>굴요리', '음식점>뷔페', '카페,디저트>다방', '한식>닭발', '육류,고기요리>닭발', '육류,고기요리>돼지고기구이', '음식점>도시락,컵밥', '한식>감자탕', '한식>해장국', '한식>기사식당', '일식>일식당', '한식>추어탕', '일식>초밥,롤', '음식점>아시아음식', '한식>백반,가정식', '한식>오징어요리', '뷔페>고기뷔페', '한식>순대,순댓국', '한식>육류,고기요리>곱창,막창,양', '해물,생선요리>복어요리', '육류,고기요리>찜닭', '해물,생선요리>게요리', '음식점>한식뷔페', '음식점>게요리', '한식>칼국수,만두', '음식점>일식,초밥뷔페', '음식점>인도음식', '육류,고기요리>정육식당', '음식점>한식>육류,고기요리>소고기구이', '음식점>양식', '육류,고기요리>족발,보쌈', '한식>해물,생선요리', '음식점>한식>백반,가정식', '한식>쌈밥', '중식>양꼬치', '한식>소고기구이', '한식>족발,보쌈', '한식>육류,고기요리', '음식점>한식>국밥', '분식>김밥', '일식>우동,소바', '음식점>일식>돈가스', '한식>국밥', '한식>닭요리', '한식>한정식', '한식>주꾸미요리', '음식점>햄버거', '음식점>분식', '음식점>브런치', '음식점>푸드코트', '중식>중식당', '한식>찜닭', '한식>생선회', '한식>백숙,삼계탕', '일식>샤브샤브', '음식점>야식', '음식점>치킨,닭강정', '한식>국수', '한식>냉면', '한식>아귀찜,해물찜', '이탈리아음식>스파게티,파스타전문', '한식>곰탕,설렁탕', '한식>곱창,막창,양', '음식점>한식>육류,고기요리>족발,보쌈', '음식점>이탈리아음식', '음식점>양갈비', '한식>찌개,전골', '음식점>퓨전음식', '양식>핫도그', '한식>매운탕,해물탕', '쇼핑,유통>가공식품', '분식>종합분식', '도시락,컵밥>다이어트,샐러드', '한식>오리요리', '음식점>피자', '음식점>밀키트', '음식점>베트남음식', '한식>돼지고기구이', '음식점>한식', '일식>일본식라면', '한식>막국수', '한식>조개요리', '분식>만두', '일식>돈가스', '양식>멕시코,남미음식', '음식점>한식>칼국수,만두', '음식점>태국음식', '음식점>패밀리레스토랑', '한식>낙지요리', '한식>닭갈비']
    cafe = ['음식점>카페,디저트>테마카페', '카페,디저트>바르바커피', '음식점>카페,디저트>베이커리>밀도', '음식점>카페,디저트>호두과자', '카페,디저트>베이커리>스마일찹쌀꽈배기', '음식점>카페,디저트>도넛', '음식점>카페,디저트>아이스크림', '카페,디저트>커피번', '가공식품>과자,사탕,초코렛', '음식점>카페,디저트>과일,주스전문점', '카페,디저트>차', '테마카페', '카페,디저트>베이커리>스마일명품찹쌀꽈배기', '음식점>카페,디저트>카페', '제조업>아이스크림,빙과류제조', '카페,디저트>테마카페', '전통식품>떡,한과', '카페,디저트>초콜릿전문점', '카페,디저트>빙수', '카페,디저트>찐빵', '쇼핑,유통>과자,사탕,초코렛', '카페,디저트>과일,주스전문점', '카페,디저트>플라워카페', '카페,디저트>호두과자', '카페,디저트>아이스크림', '카페,디저트>도넛', '카페,디저트>와플', '카페,디저트>테이크아웃커피', '음식점>카페,디저트>베이커리', '카페,디저트>베이커리>바오밥나무과자점', '룸카페', '카페,디저트>베이커리', '음식점>카페,디저트', '카페,디저트>케이크전문', '카페,디저트>카페'] 
    etc = ['패션>신발','쇼핑,유통>상가,아케이드','쇼핑,유통>패션잡화','쇼핑,유통>캐주얼웨어','쇼핑,유통>신발','쇼핑,유통>종합생활용품','서비스,산업>장소대여','여행,명소>산책로', '협회,단체>레저,스포츠',  '보드카페', '여행,명소>유원지',  '여행,명소>도립공원',  '고양이카페', '스포츠,오락>나이트클럽', '스포츠,오락>클럽',  '스포츠,오락>아이스링크',  '쇼핑,유통>쇼핑복합시설', '가구,인테리어>가구', '방탈출카페',  '슬라임카페', '스포츠시설>골프장', '여행,명소>식물원,수목원', '스포츠,오락>롤러,인라인스케이트장', '오락시설>놀이터',  '스포츠,오락>실내골프연습장', '사진,스튜디오>프로필사진전문', '협회,단체>수영', '여행,명소>템플스테이',  '생활,편의>향수공방',  '스포츠,오락>체육관', '오락시설>PC방' ,  '사진,스튜디오>앨범','스포츠,오락>야구장', '스포츠,오락>어린이축구장',  '임대,대여>만화,도서', '스포츠,오락>무도장,콜라텍', '여행,명소>먹자거리', '스포츠,오락>멀티방',  '스포츠,오락>DVD방', '스포츠,오락>스케이트장', '스포츠시설>스크린골프장', '여행,명소>거리,골목', '생활,편의>복권,로또', '도시,테마공원>부속시설', '스포츠,오락>암벽등반', '문화,예술>문화,예술인', '스포츠,오락>낚시터', '스포츠,오락>농구장', '스포츠,오락>복싱,권투장', '숙박>게스트하우스', '문화,예술>영화관', '생활,편의>찜질방', '여행,명소>관람,체험', '문화,예술>미술관', '여행,명소>박물관', '자연공원>부속시설', '생활,편의>목욕탕,사우나', '여행,명소>천문대', '여행,명소>도시,테마공원', '스포츠,오락>풋살장', '스포츠,오락>스포츠시설', '문화,예술>공연장', '여행,명소>자연,생태공원', '스포츠,오락>노래방', '교육,학문>도서관', '마사지,지압>발관리', '스포츠,오락>수영장', '도서,음반,문구>문구,팬시용품', '스포츠,오락>PC방', '키즈카페,실내놀이터', '쇼핑,유통>아울렛', '가구,인테리어>인테리어소품', '스포츠시설>헬스장', '미용>피부,체형관리', '스포츠시설>필라테스', '문화,예술>갤러리,화랑', '스포츠,오락>테니스장', '스포츠시설>골프연습장', '문화,예술>복합문화공간', '레져,체육시설>공원', '생활,편의>사진,스튜디오', '쇼핑,유통>서점', '오락시설>노래방', '미용>왁싱,제모', '스포츠,오락>당구장', '교육,학문>독서실', '생활,편의>미용실', '동물카페', '애견카페', '스포츠,레크레이션학원>권투,복싱', '스포츠,오락>탁구장', '스포츠,오락>클럽하우스', '여행,명소>근린공원', '스터디카페', '힐링카페', '쇼핑,유통>슈퍼,마트', '생활,편의>편의점', '생활,편의>공방', '스포츠,오락>오락시설', '스포츠,오락>축구장', '교육,학문>어린이도서관', '쇼핑,유통>백화점', '카페,디저트>북카페', '카페,디저트>스터디카페', '쇼핑,유통>시장', '스포츠,오락>만화방', '오락시설>만화방', '문화,예술>박람회,전시회', '생활,편의>목욕,찜질', '스포츠,오락>공설,시민운동장', '피부,체형관리>태닝', '갤러리카페', '숙박>펜션', '숙박>전통숙소', '골프장>퍼블릭골프장', '레저,테마>테마파크', '여행,명소>관광농원,팜스테이', '교육,학문>시립도서관', '여행,명소>동물원', '미용>미용실', '미용>네일아트,네일샵', '자연명소>휴양림,산림욕장', '스포츠,오락>승마장', '스포츠시설>파크골프장', '캐리비안베이 빌리지', '스포츠,오락', '반려동물>애견훈련', '문화,예술>공연,연극시설', '숙박>캠핑,야영장', '도서관>독서실', '숙박>호텔', '스포츠,오락>볼링장', '여행,명소>체험마을', '스포츠,오락>오락실', '미용>속눈썹증모,연장', '체험,공연,전시 시설', '스포츠,오락>행글라이딩,패러글라이딩', '사주카페', '스포츠,오락>양궁장', '여행,명소>테마공원', '쇼핑,유통>예술품,골동품', '쇼핑,유통>화장품,향수', '서점>독립서점', '사진,스튜디오>셀프,대여스튜디오', '여행,명소>테마파크', '여행,명소>자연공원', '휴양림,산림욕장>부속시설', '스포츠,오락>실외골프연습장', '스포츠,오락>골프장', '문화,예술>문화,예술회관', '지명>계곡,협곡', '미용>마사지,지압', '종합도소매>슈퍼,마트', '문화,예술>문화원', '사진,스튜디오>디지털인화', '도서,음반,문구>아동도서', '스포츠,오락>서바이벌게임', '스포츠,오락>사격장', '스포츠,오락>수상스키', '스포츠,오락>스키장', '스포츠,오락>ATV체험장', '스포츠,오락>레포츠시설', '레져,체육시설>운동장', '스포츠,오락>유흥시설', '쇼핑,유통>주류', '스포츠,오락>실내체육관', '패션>여성의류', '쇼핑,유통>천연화장품', '여행,명소>체험여행', '지명>저수지,제', '문화,예술>전시관', '골프장>par3골프장', '반려동물>반려동물호텔', '반려동물용품>애견용품', '스포츠,오락>스포츠센터', '스포츠,오락>궁도장', '숙박>모텔', '생활,편의>꽃집,꽃배달', '숙박>콘도,리조트', '여행,명소>촬영장소', '생활,편의>미용', '레저,테마>눈썰매장', '반려동물>반려견놀이터', '지역명소>도시,테마공원', '숙박>관광농원,팜스테이', '여행,명소>온천,스파', '스포츠,오락>골프연습장', '쇼핑,유통>도서,음반,문구', '교통,운수>공영주차장', '쇼핑,유통>식료품', '교통시설>주차장', '쇼핑,유통>문구,팬시용품', '취미,레저용품>무선모형(RC)', '스포츠,오락>배드민턴장', '캠핑장>사이트'] 
    bar = ['술집>전통,민속주점>대반전', '술집>전통,민속주점>민속주점', '술집>이자카야>잔잔',  '음식점>술집', '음식점>술집>요리주점',  '음식점>술집>오뎅,꼬치', '술집>오뎅,꼬치', '술집>이자카야', '음식점>술집>맥주,호프', '술집>전통,민속주점', '술집>요리주점', '술집>바(BAR)', '카페,디저트>라이브카페', '술집>와인', '술집>단란주점', '술집>포장마차', '술집>맥주,호프', '술집>유흥주점']

    restaurant_index = []
    cafe_index = []
    etc_index = []
    bar_index = []
    for i in range(len(result)):
        if result['category'][i] in restaurant:
            restaurant_index.append(i)
        elif result['category'][i] in cafe:
            cafe_index.append(i)
        elif result['category'][i] in etc:
            etc_index.append(i)
        elif result['category'][i] in bar:
            bar_index.append(i)
        else:
            continue

    # 음식점 추천 결과
    result_restaurant = result.loc[restaurant_index]

    # 카페 추천 결과
    result_cafe = result.loc[cafe_index]

    # 기타 추출 결과
    result_etc = result.loc[etc_index]

    # 술집 추천 결과
    result_bar = result.loc[bar_index]

    def hash_tag(df_top6, i):
        okt = Okt()

        top_1 = df_top6.iloc[i]
        
        # 가게별 리뷰 합친것 -> 다시 분리
        texts = list(map(str, top_1['review'].split('<>')))
        
        # 변환
        for j in range(len(texts)):
            texts[j] = okt.nouns(texts[j]) # 명사
            # texts[j] = okt.morphs(texts[j]) # 말뭉치
            texts[j] = ' '.join(texts[j])
        
        # 한글, 영어, 숫자 제외한 다른 글자 제거
        texts = [normalize(text, english=True, number=True) for text in texts]
        

        min_count = 1 # 단어의 최소 출현 빈도수 (그래프 생성 시)
        max_length = 10 # 단어의 최대 길이
        wordrank_extractor = KRWordRank(min_count, max_length, verbose=False, )

        beta = 0.85 # PageRank의 decaying factor beta
        max_iter = 10

        keywords, rank, graph = wordrank_extractor.extract(texts, beta, max_iter)
        
        hash_tag = []
        for word, r in sorted(keywords.items(), key=lambda x:x[1], reverse=True):
            hash_tag.append(word)
        
        # 불용어 제거
        for stop in stopwords:
            for tag in hash_tag:
                if stop == tag:
                    hash_tag.remove(stop)            
        
        # 해시태그 결과 - 4개
        tag_str = ''
        hash_tag = hash_tag[:4]
        for tag in hash_tag:
            tag_str += f'#{tag} '
        
        return top_1['name'], tag_str, keywords
        
        
    def make_wc(keywords, mask):
    # 폰트 위치 지정
        font_path = 'chat/static/fonts/BMHANNAPro.ttf'

        krwordrank_cloud = WordCloud(
            font_path = font_path,
            width = 300,
            height = 200,
            max_font_size = 100,
            background_color="white",
            max_words = 20,
            stopwords=stopwords,
            mask = mask,
            colormap='GnBu_r',     
        )

        krwordrank_cloud = krwordrank_cloud.generate_from_frequencies(keywords)

        return krwordrank_cloud

        
        
    def show_recommend(rec_df):
        # 추천장소 3개 데이터프레임
        df_top6 = rec_df.iloc[:3].copy()

        # 워드클라우드에 사용할 모양 불러오기
        mask = np.array(Image.open(mask_path))
        lists = []
        # 해시태그와 이미지 출력
        for i in range(len(df_top6)):
            name, tag, keywords = hash_tag(df_top6, i)
            krwordrank_cloud = make_wc(keywords, mask)
            lists.append(krwordrank_cloud)
        return [lists, df_top6]
            
    return_value = []
    return_value.append(show_recommend(result_restaurant))
    return_value.append(show_recommend(result_cafe))
    return_value.append(show_recommend(result_bar))
    return_value.append(show_recommend(result_etc))
    return return_value