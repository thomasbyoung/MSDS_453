# (Second Script - LLaMA Integration)

import json
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, List, Tuple
import textwrap
from dotenv import load_dotenv
load_dotenv()

class LlamaChatbot:
    def __init__(self):
        self.model_path = os.getenv("MODEL_PATH")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        print("Loading model with 4-bit quantization...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            max_memory={0: "8GB"},  
            offload_folder="offload"  
        )
        
        print("Loading document base...")
        self.load_documents()
        
    def generate_response(self, query: str) -> Dict:

        prompt = f"""<s>[INST] You are a helpful AI assistant for Clune Construction. 
        Answer this question concisely: {query} [/INST]</s>"""

        print("Tokenizing input...")
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=512  
        )
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            
        print("Generating response...")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,  
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,  
                use_cache=True,  
                num_beams=1  
            )
        
        print("Decoding response...")
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        

        try:
            response = response.split("[/INST]")[-1].strip()
        except:
            response = response.strip()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return {
            "query": query,
            "response": response,
            "relevant_documents": []
        }

    def load_documents(self):

        try:
            with open("processed_user_guides.json", 'r', encoding='utf-8') as f:
                self.docs = json.load(f)
            print(f"Loaded {len(self.docs)} documents")
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            raise

def interactive_mode(chatbot):
    print("\nEntering interactive mode. Type 'quit' to exit.")
    print("GPU Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory // 1024**2, "MB")
        print("Initial GPU Memory Used:", torch.cuda.memory_allocated() // 1024**2, "MB")
    
    while True:
        query = input("\nEnter your question: ").strip()
        
        if query.lower() == 'quit':
            break
            
        try:
            print("Processing query...")
            response = chatbot.generate_response(query)
            print("\nResponse:", response['response'])
            
            if torch.cuda.is_available():
                print("GPU Memory Used:", torch.cuda.memory_allocated() // 1024**2, "MB")
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            if torch.cuda.is_available():
                print("GPU Memory Used:", torch.cuda.memory_allocated() // 1024**2, "MB")

def main():
    print("Initializing Clune Construction Assistant with LLaMA 3...")
    try:
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
            print("Total GPU Memory:", torch.cuda.get_device_properties(0).total_memory // 1024**2, "MB")
        
        chatbot = LlamaChatbot()
        interactive_mode(chatbot)
    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")
        return

if __name__ == "__main__":
    main()