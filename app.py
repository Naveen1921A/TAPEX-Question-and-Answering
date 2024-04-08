

import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, BartForConditionalGeneration

# Load the TAPEX tokenizer and model (replace with your fine-tuned model names)
tokenizer = AutoTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-large-finetuned-wtq")


def predict(table_path, query):
  """
  Predicts answer to a question using the TAPEX model on a given table.

  Args:
      table_path: Path to the CSV file containing the table data.
      query: The question to be answered.

  Returns:
      The predicted answer as a string.
  """
  # Load the sales data from CSV
  sales_record = pd.read_csv(r"C:\Users\Windows\Desktop\Main Projects\Efficient Inventory Management With LLM\2000 Rows Dataset.csv")
  sales_record = sales_record.astype(str)  # Ensure string type for tokenizer

  # Truncate the input to fit within the model's maximum sequence length
  max_length = model.config.max_position_embeddings
  encoding = tokenizer(table=sales_record, query=query, return_tensors="pt", truncation=True, max_length=max_length)

  # Generate the output
  outputs = model.generate(**encoding)

  # Decode the output
  prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
  return prediction

st.title("TAPEX Table Q&A App")

# Upload table data
uploaded_file = st.file_uploader("Upload Sales Data (CSV)", type="csv")

if uploaded_file is not None:
  # Read the uploaded CSV file
  df = pd.read_csv(uploaded_file)
  st.write(df)  # Display the uploaded table

  # User query input
  query = st.text_input("Ask a question about the sales data:")

  if query:
    # Predict answer using the model
    prediction = predict(uploaded_file.name, query)
    st.write(f"**Your Question:** {query}")
    st.write(f"**Predicted Answer:** {prediction}")
else:
  st.info("Please upload a CSV file containing your sales data.")


