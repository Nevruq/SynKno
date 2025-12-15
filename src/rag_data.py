import json
import re
from pydantic import BaseModel


class ragData:
    #Question, Ideal_answer, reponse_1{metadaten}, response_2{metadaten}, comparission{metadaten}
    def __init__(self, question=None, ideal_answer=None, 
                 query_input_1=None, query_output_1=None, LLM_answer_method_1=None,
                 query_input_2=None, query_output_2=None, LLM_answer_method_2=None,
                  comparisson=None):

        self.question = question
        self.ideal_answer = ideal_answer,
        self.query_input_1=query_input_1, 
        self.query_output_1=query_output_1, 
        self.LLM_answer_method_1=LLM_answer_method_1,
        self.query_input_2=query_input_2, 
        self.query_output_2=query_output_2, 
        self.LLM_answer_method_2=LLM_answer_method_2,
        self.comparisson = comparisson

    def create_json(self):
        return {"question": self.question,
                "ideal_answer": self.ideal_answer,
                "method_1": {
                    "query_input": "none",
                    "query_output": "none",
                    "LLM_answer": "none"
                    },
                "method_2": {
                        "query_input": "none",
                        "query_output": "none",
                        "LLM_answer": "none"
                    },
                "comparisson": self.comparisson
                }


# Beispiel für das geschachtelte Modell (falls Sie es verwenden)
# class MethodData(BaseModel):
#     queries: dict = {}
#     LLM_answer: str = "none"

    # Definieren Sie hier Ihre Felder, wie in den Attributen Ihrer __init__
    question: str
    ideal_answer: list  # Beibehalten als Liste, falls es so im Pydantic-Modell ist
    
    # Methoden-Felder
    method_1: dict 
    method_2: dict
    
    # Vergleichs-Feld (kann ein Pydantic-Modell oder ein String sein, wie im Beispiel)
    comparisson: str 

    # --- NEUE FUNKTIONEN START ---

    @staticmethod
    def get_csv_header() -> str:
        """Gibt die Semikolon-separierte CSV-Kopfzeile zurück."""
        return "question;ideal_answer;LLM_answer_method_1;LLM_answer_method_2;comparison_explanation"

    def _clean_csv_value(self, value):
        """Entfernt Zeilenumbrüche und maskiert Anführungszeichen für CSV."""
        if not isinstance(value, str):
            value = str(value)
            
        # Entferne Zeilenumbrüche
        value = re.sub(r'[\n\r]+', ' ', value).strip()
        
        # Maskiere doppelte Anführungszeichen (ersetze " durch "")
        value = value.replace('"', '""')
        
        # Umschließe den Wert mit Anführungszeichen, falls er das Trennzeichen enthält
        if ';' in value or ',' in value: # Füge auch Kommas hinzu, falls Sie später zu Komma wechseln
             return f'"{value}"'
        
        return value

    def to_csv_string(self) -> str:
        """
        Konvertiert die Klassenattribute in einen Semikolon-separierten CSV-String.
        
        Returns:
            Ein String im CSV-Format, getrennt durch ';'.
        """
        
        # 1. Datenextraktion und Normalisierung
        
        # ideal_answer: Nimmt den ersten String aus der Liste/Tupel
        ideal_ans = self.ideal_answer[0] if self.ideal_answer and isinstance(self.ideal_answer, (list, tuple)) else ""
        
        # LLM Antworten (nimmt an, sie sind unter 'LLM_answer' im Dict gespeichert)
        llm_ans_1 = self.method_1.get("LLM_answer", "")
        llm_ans_2 = self.method_2.get("LLM_answer", "")
        
        # Vergleich: Parsen des JSON-Strings
        explanation = ""
        try:
            # Annahme: self.comparisson ist ein String, der JSON enthält
            if isinstance(self.comparisson, str):
                comparisson_data = json.loads(self.comparisson)
            else:
                 # Falls es bereits ein Dict ist (z.B. durch ein ParsedResponse)
                comparisson_data = self.comparisson
                
            explanation = comparisson_data.get("explanation", "")
            
        except (json.JSONDecodeError, AttributeError):
            explanation = "Comparison data could not be parsed."

        
        # 2. Werte bereinigen und in die Liste einfügen
        cleaned_values = [
            self._clean_csv_value(self.question),
            self._clean_csv_value(ideal_ans),
            self._clean_csv_value(llm_ans_1),
            self._clean_csv_value(llm_ans_2),
            self._clean_csv_value(explanation)
        ]
        
        # 3. CSV-String erstellen
        return ";".join(cleaned_values)

# ... (Ende der Klasse)