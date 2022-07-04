# Load the model in python
from fairseq.models.transformer import TransformerModel

# model = TransformerModel.from_pretrained(
#     'tr_models/text/WMT_de-en/',
#     checkpoint_file='checkpoint_best.pt',
#     target_dict='dict.en.txt',
#     target_spm='spm.model',
#     src='de'
# )
# model.eval()  # disable dropout (or leave in train mode to finetune)
#
# # Translate
# model.translate("Hallo.")
# # model.translate(["Hallo.", "Hallo Welt."])

# Fairseq pre-trained text-model

model = TransformerModel.from_pretrained(
    'tr_models/wmt16.en-de.joined-dict.transformer/',
    checkpoint_file='model.pt',
    src='de',
    target_dict='dict.en.txt',
    target_spm='spm.model',
    bpe='fastbpe'
)
model.eval()  # disable dropout (or leave in train mode to finetune)

# Translate
translation = model.translate("Mein Name ist Anastasia.")
print('pre-trained text-model from fairseq:', translation)
