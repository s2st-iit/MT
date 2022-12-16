import ctranslate2
import sentencepiece as spm
from ctranslate2 import translator

# Set file paths
source_file_path = "test.hi"
target_file_path = "test.en"

sp_source_model_path = "vocab.src.model"
sp_target_model_path = "vocab.tgt.model"

ct_model_path = "model/"


# Load the source SentecePiece model
sp = spm.SentencePieceProcessor()
sp.load(sp_source_model_path)

# Open the source file
with open(source_file_path, "r") as source:
  lines = source.readlines()

source_sents = [line.strip() for line in lines]

# Subword the source sentences
source_sents_subworded = sp.encode(source_sents, out_type="str")

# Translate the source sentences
translator = ctranslate2.Translator(ct_model_path, device="cpu")  # or "cuda" for GPU
translations = translator.translate_batch(source_sents_subworded, batch_type="tokens", max_batch_size=4096)
translations = [translation[0]['tokens'] for translation in translations]

# Load the target SentecePiece model
sp.load(sp_target_model_path)

# Desubword the target sentences
translations_desubword = sp.decode(translations)


# Save the translations to the a file
with open(target_file_path, "w+", encoding="utf-8") as target:
  for line in translations_desubword:
    target.write(line.strip() + "\n")

print("Done")
