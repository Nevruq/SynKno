#import wikipedia
from transformers import BertTokenizer, BertModel, AutoTokenizer
import torch
import cBert.wrapper_CBert as cBert
import weighting_prompt as wp
import chromadb_handler as ch
import call_llm
import json
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
   prompt = "How did American Airlines (AAL) describe its financial outlook for the second half of 2023, and what factors did the company highlight as influencing its performance?"
   ideal_answer = """
                  American Airlines stated that its financial outlook for the second half of 2023 remained cautiously positive, supported by strong travel demand and improved operational efficiency. The company highlighted rising fuel costs, labor-related expenses, and capacity constraints as key factors influencing its performance.
                  Management also pointed out that continued recovery in international routes and steady corporate travel were expected to contribute to revenue stability.
   """
   # 2. Frage wird gestillt und gechunked
   response_categories = call_llm.call_llm_categories(prompt, LLM_client).output_text
   response_categories_json = json.loads(response_categories)
   print(response_categories, type(response_categories))
   output_file = open("out.txt", "w", encoding="utf-8")
   output_file.write(response_categories)
   #destilled_response = call_llm.call_llm_categories(prompt, model)
   #formated_destilled_response = wp.distill_query(user_prompt=prompt, raw_answer=destilled_response)
   #3 connect to client
   client = chromadb.PersistentClient("src/chroma")
   collection = client.get_or_create_collection("nasdaq_news")

   # 3a. Query Datenbank mit Chunking. JSON -> List
   #formated_subqueries = [query for query in formated_destilled_response.items()]
   #query_destilled_rag_return = collection.query(query_texts="formated text", n_results=3)
   topics_formatted = [response_categories_json["subQuerie1"], response_categories_json["subQuerie2"], response_categories_json["subQuerie3"]]
   query_summary_results = collection.query(query_texts=topics_formatted, n_results=3)["documents"][0]
   
   # 3b. Query Datenbank ohne Chunking
   query_regular_rag = collection.query(query_texts=prompt, n_results=3)
   context_query = query_regular_rag["documents"][0]
   asnwer_llm_wRAG = call_llm.call_llm_answer_RAG(query_results=context_query, user_prompt=prompt, model=LLM_client)
   # Normales Rag mit suche nach ganzer Prompt

   # Destillierte Themen

   answer_llm_Chunked_Rag = call_llm.call_llm_answer_RAG(query_results=query_summary_results, user_prompt=prompt, model=LLM_client)
   # 4. Vergleiche beide Antworten mit musterantwort einer LLM
   # = call_llm.call_llm_regular(user_prompt=prompt, model=LLM_client)
   output_file = open("output_LLM.txt", "w", encoding="utf-8")
   output_file.write(f"Answe wRag: \n {asnwer_llm_wRAG} \n Answer withoutRag: \n {answer_llm_Chunked_Rag}", )
   response_compare = call_llm.call_llm_compare_answers(prompt=prompt,
                                                        ideal_answer=ideal_answer,
                                                        answer_destilled=asnwer_llm_wRAG.output_text, 
                                                        answer_regular=answer_llm_Chunked_Rag.output_text,
                                                        model=LLM_client)
   output_file.write("\n Comparrison of both Answers: \n" + response_compare.output_text)
   #print(call_llm.call_llm_compare_answers())