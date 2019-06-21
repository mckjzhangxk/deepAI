from recognize.ccptUtils import CCPD_Recognizer
import torch

def testRecognize():
    device=torch.device('cuda')
    detector=CCPD_Recognizer(device=device)

