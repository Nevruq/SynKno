#import wikipedia
from transformers import BertTokenizer, BertModel, AutoTokenizer
import torch
import cBert.wrapper_CBert as cBert
import weighting_prompt as wp
import chromadb_handler as ch
import call_llm
import json
import pandas as pd
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
   
   # Open output file
   output_file = open("output_data_LLM.txt", "w", encoding="utf-8")
   # write header
   output_file.write("Question, Ideal_answer, reponse_1, response_2, comparisson")
   client = chromadb.PersistentClient("src/chroma")
   collection = client.get_or_create_collection("nasdaq_news_chunked")

   QA_nasdaq = pd.read_csv("QA_nasdaq")

   for row in QA_nasdaq.itertuples(index=False):
      response_1 = dict()
      response_2 = dict()
      # 2. Frage wird gestillt und gechunked
      response_categories = call_llm.call_llm_categories(row.question, LLM_client).output_text
      response_categories_json = json.loads(response_categories)


      #destilled_response = call_llm.call_llm_categories(prompt, model)
      #formated_destilled_response = wp.distill_query(user_prompt=prompt, raw_answer=destilled_response)
      #3 connect to client
      client = chromadb.PersistentClient("src/chroma")
      collection = client.get_or_create_collection("nasdaq_news_chunked")

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
      print(context_query)
      print(query_summary_results)
      # 4. Vergleiche beide Antworten mit musterantwort einer LLM
      # = call_llm.call_llm_regular(user_prompt=prompt, model=LLM_client)
      
      output_file.write(f"Answe wRag: \n {asnwer_llm_wRAG} \n Answer withoutRag: \n {answer_llm_Chunked_Rag}", )
      response_compare = call_llm.call_llm_compare_answers(prompt=prompt,
                                                         ideal_answer=ideal_answer,
                                                         answer_destilled=asnwer_llm_wRAG.output_text, 
                                                         answer_regular=answer_llm_Chunked_Rag.output_text,
                                                         model=LLM_client)
      output_file.write("\n Comparrison of both Answers: \n" + response_compare.output_text)
      #print(call_llm.call_llm_compare_answers())

output_file.close