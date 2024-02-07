from fastai.vision.all import *
from fastai.vision.widgets import *
path = Path()
path.ls(file_exts='.pkl')
learn_inf = load_learner(path/'export.pkl')
labels = learn_inf.dls.vocab
def on_click_classify(image):
     img = PILImage.create(image)
     pred,pred_idx,probs = learn_inf.predict(img)
     return {labels[i]: float(probs[i]) for i in range(len(labels))}
import gradio as gr
title = "Classify a Bear! Check if it is a grizzly, black or teddy bear! "
gr.Interface(fn=on_click_classify, inputs=gr.Image(), outputs=gr.Label(num_top_classes=3), title=title).launch(share=True)