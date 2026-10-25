"""Diagnostic classes used by the 30-class laryngoscopy model.

Classes 0–6 are malignant. Their probabilities sum to P(malignant).
"""

CLASS_NAMES = (
    "Epi_Mal",
    "VC_Mal",
    "FVC_Mal",
    "BOT_Mal",
    "NP_Mal",
    "Hypo_Mal",
    "OP_Mal",
    "Lary_Pap",
    "OP_Pap",
    "Hypo_Pap",
    "Hema",
    "NP_Cyst",
    "Epi_Cyst",
    "BOT_Cyst",
    "VC_Cyst",
    "Hypo_Cyst",
    "NP_LH",
    "VC_Pol",
    "VP_Gran",
    "VC_Nod",
    "VC_Inf",
    "VC_RE",
    "Lary_SP",
    "Hypo_SP",
    "NC_Norm",
    "NP_Norm",
    "Lary_Norm",
    "OP_Norm",
    "BOT_Norm",
    "PS_Norm",
)

N_CLASSES = len(CLASS_NAMES)
N_MALIGNANT = 7
MALIGNANT_THRESHOLD = 0.5
