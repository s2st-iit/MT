Step 1: Preprocessing the data

1. Download the parallel corpus from Samanantar or Opus and extract them to a folder
2. Run DataPreprocessing.py inside Model folder(This will generate two new files after preprocessing)
3. Currently the preprocessing that is being carried out by the code are as follows:
   1. Converting all the capital letters to small letters.
   2. Removing non-printable characters.
   3. Reoving emojis.
   4. Removing unknown characters (Heavily dataset dependent)
4.We will be using sentencepiece as the tokenizer

Step 2: Training the model
1. Currently there are three models CNNSeq2Seq, Transformer model and RNN based LSTM network.
2. Just feed the data generated from step 1. into the model.

Step 3: Inference
1. Run translate.py after changing the source and target file path also change the vocabularies file path
