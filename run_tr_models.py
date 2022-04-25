# Load the model in python
from fairseq.models.visual import VisualTextTransformerModel
model = VisualTextTransformerModel.from_pretrained(
    checkpoint_file='tr_models/WMT_de-en/checkpoint_best.pt',
    target_dict='tr_models/WMT_de-en/dict.en.txt',
    target_spm='tr_models/WMT_de-en/spm.model',
    src='de',
    image_font_path='fairseq/data/visual/fonts/NotoSans-Regular.ttf'
)
model.eval()  # disable dropout (or leave in train mode to finetune)

# Translate
print(model.translate("Mein Name ist Anastasia."))
