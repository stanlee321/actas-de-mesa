"""
Class for return the used labels.
"""

class Labels:
    def __init__(self):
        self.name_maping = {
                    1: "Votos Nulos Presidente",
                    2: "Votos Nulos Diputado",
                    3: "Votos Blancos Presidente",
                    4: "Votos Blancos Diputado",
                    5: "Votos Validos Presidente",
                    6: "Votos Validos Diputado",
                    7: "PAN-BOL Presidente",
                    8: "PAN-BOL Diputado",
                    9: "MNR Presidente",
                    10: "MNR Diputado",
                    11: "PDC Presidente",
                    12: "PDC Diputado",
                    13: "21F Presidente",
                    14: "21F Diputado ",
                    15: "MAS IPSP Presidente",
                    16: "MAS IPSP Diputado",
                    17: "UCS Presidente" ,
                    18: "UCS Diputado",
                    19: "MTS Presidente",
                    20: "MTS Diputado",
                    21: "FPV Presidente",
                    22: "FPV Diputado",
                    23: "CC Presidente",
                    24: "CC Diputado"
                }
        self.name_maping_results = {
                    25: "Votos Nulos Presidente, Result",
                    26: "Votos Nulos Diputado, Result",
                    27: "Votos Blancos Presidente, Result",
                    28: "Votos Blancos Diputado, Result",
                    29: "Votos Validos Presidente, Result",
                    30: "Votos Validos Diputado, Result",
                    31: "PAN-BOL Presidente, Result",
                    32: "PAN-BOL Diputado, Result",
                    33: "MNR Presidente, Result",
                    34: "MNR Diputado, Result",
                    35: "PDC Presidente, Result",
                    36: "PDC Diputado, Result",
                    37: "21F Presidente, Result",
                    38: "21F Diputado, Result",
                    39: "MAS IPSP Presidente, Result",
                    40: "MAS IPSP Diputado, Result",
                    41: "UCS Presidente, Result" ,
                    42: "UCS Diputado, Result",
                    43: "MTS Presidente, Result",
                    44: "MTS Diputado, Result",
                    45: "FPV Presidente, Result",
                    46: "FPV Diputado, Result",
                    47: "CC Presidente, Result",
                    48: "CC Diputado, Result"
                }

    def get_name_mapings(self):
        return self.name_maping
  
    def get_name_mapings_results(self):
        return self.name_maping_results

    def get_full_data(self):

        return {**self.get_name_mapings(), **self.get_name_mapings_results()}

    def create_attrDict(self):
        items = self.get_full_data()
        item_list = []

        for k,v in items.items():
            item_list.append({
                "supercategory": "none", "id": k, "name": v
            })

        return item_list

if __name__ == "__main__":

    labels = Labels()

    items_list = labels.create_attrDict()

    print(items_list)