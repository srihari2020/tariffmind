import re
import pandas as pd


class HSValidator:
    def __init__(self, csv_path: str = "data/hs_2022.csv"):
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        df["level"] = df["level"].astype(int)

        self.chapters = df[df["level"] == 2].set_index("hscode")["description"].to_dict()
        self.headings = df[df["level"] == 4].set_index("hscode")["description"].to_dict()
        self.subheadings = df[df["level"] == 6].set_index("hscode")["description"].to_dict()
        self.sections = df[df["level"] == 6].set_index("hscode")["section"].to_dict()

    def normalize(self, raw_code: str) -> str:
        return re.sub(r"[^0-9]", "", str(raw_code))

    def validate(self, raw_code: str) -> dict:
        code = self.normalize(raw_code)
        if len(code) != 6 or code not in self.subheadings:
            return {
                "is_valid": False,
                "hs_code_6digit": code,
                "error": f"Invalid or non-existent 6-digit HS code: '{raw_code}'",
                "hierarchy": None,
            }

        ch_code, hd_code = code[:2], code[:4]
        return {
            "is_valid": True,
            "hs_code_6digit": code,
            "error": None,
            "hierarchy": {
                "section": self.sections[code],
                "chapter_2": f"{ch_code} - {self.chapters.get(ch_code, '')}",
                "heading_4": f"{hd_code} - {self.headings.get(hd_code, '')}",
                "subheading_6": f"{code} - {self.subheadings[code]}",
            },
        }


if __name__ == "__main__":
    v = HSValidator()
    print("Valid test:", v.validate("6201.40"))
    print("Hallucination test:", v.validate("999999"))
