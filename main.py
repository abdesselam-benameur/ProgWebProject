
import pandas as pd

filepath = "data/dermatology.data"

df = pd.read_csv(filepath, header=None)
df.head()
# set the list of columns of df
df.columns = ["erythema", "scaling", "definite_borders", "itching", "koebner_phenomenon",
              "polygonal_papules", "follicular_papules", "oral_mucosal_involvement",
              "knee_and_elbow_involvement", "scalp_involvement", "family_history",
              "melanin_incontinence", "eosinophils_in_the_infiltrate", "PNL_infiltrate",
              "fibrosis_of_the_papillary_dermis", "exocytosis", "acanthosis", "hyperkeratosis",
              "parakeratosis", "clubbing_of_the_rete_ridges", "elongation_of_the_rete_ridges",
              "thinning_of_the_suprapapillary_epidermis", "spongiform_pustule", "munro_microabcess",
              "focal_hypergranulosis", "disappearance_of_the_granular_layer", "vacuolisation_and_damage_of_basal_layer",
              "spongiosis", "saw-tooth_appearance_of_retes", "follicular_horn_plug", "perifollicular_parakeratosis",
              "inflammatory_monoluclear_inflitrate", "band-like_infiltrate", "Age", "class"]

