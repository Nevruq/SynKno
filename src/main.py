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
import rag_data



if __name__ == "__main__":
   # init model
   MODEL_NAME_2 = "gpt-5-nano-2025-08-07"
   MODEL_NAME = "gpt-3.5-turbo-0125"
   load_dotenv()
   LLM_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
   # 1. Stelle eine Frage 
   
   # Open output file
   output_file = open("output_data_LLM.txt", "w", encoding="utf-8")
   # write header: Question, Ideal_answer, reponse_1{metadaten}, response_2{metadaten}, comparission{metadaten}

   client = chromadb.PersistentClient("src/chroma")
   collection = client.get_or_create_collection("nasdaq_news_chunked")

   QA_nasdaq = pd.read_csv("src/QA_nasdaq.csv", sep=",")

   for row in QA_nasdaq.itertuples(index=False):
      question = row.question
      ideal_answer = row.answer
      
      # Method 1: topics_formatted, query_summary_results, answer_llm_chunked_rag
      response_categories = call_llm.call_llm_categories(row.question, LLM_client)
      response_categories_json = json.loads(response_categories.output_text)

      topics_formatted = [response_categories_json["subQuerie1"], response_categories_json["subQuerie2"], response_categories_json["subQuerie3"]]
      query_summary_results = collection.query(query_texts=topics_formatted, n_results=3)
      query_summary_results_formatted = query_summary_results["documents"][0]

      answer_llm_Chunked_Rag = call_llm.call_llm_answer_RAG(query_results=query_summary_results, user_prompt=question, model=LLM_client)

      # Method 2: Question, query_regular_rag,  answer_llm_wRag
      query_regular_rag = collection.query(query_texts=question, n_results=3)
      context_query = query_regular_rag["documents"][0]
      asnwer_llm_wRAG = call_llm.call_llm_answer_RAG(query_results=context_query, user_prompt=question, model=LLM_client)

      # Comparisson
      response_compare = call_llm.call_llm_compare_answers(prompt=question,
                                                         ideal_answer=ideal_answer,
                                                         answer_destilled=answer_llm_Chunked_Rag.output_text, 
                                                         answer_regular=asnwer_llm_wRAG.output_text,
                                                         model=LLM_client)
      # create json with data class
      data_Json = rag_data(question=question, ideal_answer=ideal_answer, 
                           query_input_1=topics_formatted, query_output_1=query_summary_results, LLM_answer_method_1=answer_llm_Chunked_Rag,
                           query_input_2=question, query_output_2=query_regular_rag, LLM_answer_method_2=asnwer_llm_wRAG,
                           comparisson=response_compare)
      print(data_Json)
output_file.close