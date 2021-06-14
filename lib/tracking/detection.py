import numpy as np


class Detection(object):
    def __init__(self, box, score, embedding):
        self.box = box.detach()
        self.score = score
        self.embedding = embedding
        self.brand_new = True

    @property
    def ltwh(self):
        # Retrieve left, top, width, height from box (x1y1x2y2)
        ltwh = np.asarray(self.box.clone().detach().cpu().numpy())
        ltwh[2:] -= ltwh[:2]
        return ltwh

    @property
    def avg_embedding(self):
        return self.embedding.mean(-1).mean(-1)
