#import wikipedia
from transformers import BertTokenizer, BertModel, AutoTokenizer
import torch
import cBert.wrapper_CBert as cBert
import weighting_prompt as wp
import chromadb_handler as ch
import call_llm
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os



if __name__ == "__main__":
   # init model
   MODEL_NAME_2 = "gpt-5-nano-2025-08-07"
   MODEL_NAME = "gpt-3.5-turbo-0125"
   load_dotenv()
   LLM_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
   # 1. Stelle eine Frage 
   prompt = "What steps did the UK government take in 2024 to address shortages in the national healthcare workforce?"
   ideal_answer = """
                  In 2024 announced measures, the UK government expanded medical training capacity, introduced accelerated recruitment programs for international healthcare professionals, and increased funding to retain existing NHS staff.
T                 he plan also included new incentive schemes for nurses and general practitioners to work in underserved regions, along with investments in digital health infrastructure to reduce administrative workload.
                  """
   # 2. Frage wird gestillt und gechunked
   #destilled_response = call_llm.call_llm_categories(prompt, model)
   #formated_destilled_response = wp.distill_query(user_prompt=prompt, raw_answer=destilled_response)
   #3 connect to client
   client = chromadb.PersistentClient("src/chroma")
   collection = client.get_or_create_collection("bbc_news")

   # 3a. Query Datenbank mit Chunking. JSON -> List
   #formated_subqueries = [query for query in formated_destilled_response.items()]
   #query_destilled_rag_return = collection.query(query_texts="formated text", n_results=3)
   # 3b. Query Datenbank ohne Chunking
   query_regular_rag = collection.query(query_texts=prompt, n_results=3)
   print(query_regular_rag)
   asnwer_llm_wRAG = call_llm.call_llm_answer_RAG(query_results=query_regular_rag, user_prompt=prompt, model=LLM_client)
   # 4. Vergleiche beide Antworten mit musterantwort einer LLM
   answer_llm_noRag = call_llm.call_llm_regular(user_prompt=prompt, model=LLM_client)
   output_file = open("output_LLM.txt", "w", encoding="utf-8")
   output_file.write(f"Answe wRag: \n {asnwer_llm_wRAG} \n Answer withoutRag: \n {answer_llm_noRag}")
   response_compare = call_llm.call_llm_compare_answers(prompt=prompt,
                                                        ideal_answer=ideal_answer,
                                                        answer_destilled=asnwer_llm_wRAG.output_text, 
                                                        answer_regular=answer_llm_noRag.output_text,
                                                        model=LLM_client)
   output_file.write("\n Comparrison of both Answers: \n" + response_compare.output_text)
   #print(call_llm.call_llm_compare_answers())