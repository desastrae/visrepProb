# Load the model in python
from fairseq.models.visual import VisualTextTransformerModel
import h5py
import numpy as np


def convert_and_save(enc_tensor):
    pass
    counter = 0
    enc_np = enc_tensor.cpu().detach().numpy()
    with h5py.File("mytestfile.hdf5", "a") as hf:
        counter += 1
        hf.create_dataset("sentence_" + str(counter), data=enc_np, compression="gzip")


model = VisualTextTransformerModel.from_pretrained(
    checkpoint_file='tr_models/WMT_de-en/checkpoint_best.pt',
    target_dict='tr_models/WMT_de-en/dict.en.txt',
    target_spm='tr_models/WMT_de-en/spm.model',
    src='de',
    image_font_path='fairseq/data/visual/fonts/NotoSans-Regular.ttf'
)
model.eval()  # disable dropout (or leave in train mode to finetune)

# Clear encodings file
if True:
    with h5py.File("mytestfile.hdf5", "w") as hf:
        pass

# Translate
with open('sentences.txt', 'r') as f2:
    for sent in f2:
        # translation, out_enc_out = model.translate("Mein Name ist Anastasia.")
        translation, out_enc_out = model.translate(sent)

        print(translation)
        print(out_enc_out)

        convert_and_save(out_enc_out)

