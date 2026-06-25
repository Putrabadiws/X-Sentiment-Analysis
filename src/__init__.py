"""Analisis Sentimen X — sentiment analysis pipeline for Indonesian tweets.

Stages (each a module here):
    crawling      -> ambil tweet (opsional, via twikit + cookie sesi X)
    preprocessing -> clean / tokenize / slang / stopword / stemming
    labelling     -> lexicon-based labelling (positive/negative/neutral)
    modeling      -> TF-IDF + SVM (SMOTE) train / evaluate / predict
"""
