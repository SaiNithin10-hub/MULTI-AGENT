from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Load environment variables
load_dotenv()
llm = Groq(id="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

# Schema description
schema_info = """
The database consists of the following tables:

1. **Airports**: 
    - airport_code (Primary Key, String)
    - city (String)
    - country (String)
    - name (String)

2. **Flights**: 
    - flight_id (Primary Key, Integer)
    - airline (String)
    - source (ForeignKey -> Airports.airport_code, String)
    - destination (ForeignKey -> Airports.airport_code, String)
    - departure_time (DateTime)
    - arrival_time (DateTime)
    - price (Float)
    - seats_available (Integer)

3. **Passengers**: 
    - passenger_id (Primary Key, Integer)
    - name (String)
    - email (String)
    - phone (String)

4. **Bookings**: 
    - booking_id (Primary Key, Integer)
    - passenger_id (ForeignKey -> Passengers.passenger_id, Integer)
    - flight_id (ForeignKey -> Flights.flight_id, Integer)
    - booking_date (DateTime, default current time)
    - status (String)
"""

# --- AGENTS ---

# Validation Agent
validation_agent = Agent(
    name="Prompt Validator",
    role="Check if user prompt is relevant to flight booking database schema",
    model=llm,
    instructions=[
        "You are responsible for checking if a user's question is related to the flight booking database described below.",
        f"Here is the schema:\n{schema_info}",
        "If the user's prompt is NOT relevant to the schema, respond with exactly: INVALID",
        "If the prompt is relevant, respond with exactly: VALID"
    ]
)

# Query Agent
query_agent = Agent(
    name="SQL Query Generator",
    role="Generate SQL queries based on natural language and schema",
    model=llm,
    instructions=[
        f"Use the following schema to generate SQL queries:\n{schema_info}",
        "Always provide SQL in a code block using ```sql and ```. Explain briefly after the SQL.",
    ],
    markdown=True,
)

# Summarizer Agent
summarizer_agent = Agent(
    name="Query Result Summarizer",
    role="Summarize the output of SQL queries in a human-readable format based on the original user prompt and schema",
    model=llm,
    instructions=[
        "You are a helpful assistant that summarizes SQL query results.",
        f"Here is the schema to help understand the query structure:\n{schema_info}",
        "You will be given the user's original question and the results returned from the database.",
        "Respond with a concise and informative summary in plain language."
    ]
)

# --- SQLAlchemy Setup ---
engine = create_engine('sqlite:///flights.db')  # Adjust for your DB
Session = sessionmaker(bind=engine)
session = Session()

# --- Helpers ---
def validate_prompt(prompt):
    result = validation_agent.run(prompt)
    result = getattr(result, "content", str(result)).strip().upper()
    return result == "VALID"

def extract_sql_from_response(response):
    try:
        if "```sql" in response:
            return response.split("```sql")[1].split("```")[0].strip()
        elif "```" in response:
            sql = response.split("```")[1].strip()
            if sql.upper().startswith(("SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE")):
                return sql

        lines = response.split('\n')
        sql_lines, capturing = [], False
        for line in lines:
            line = line.strip()
            if not capturing and any(line.upper().startswith(kw) for kw in ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE"]):
                capturing = True
                sql_lines.append(line)
            elif capturing:
                if not line or line.startswith(("Explanation", "This query")):
                    break
                sql_lines.append(line)
        return '\n'.join(sql_lines) if sql_lines else None
    except Exception as e:
        print(f"Extraction Error: {str(e)}")
        return None

def generate_sql(prompt):
    try:
        response = query_agent.run(f"{prompt}\n\nSchema:\n{schema_info}")
        response = getattr(response, "content", str(response))
        sql = extract_sql_from_response(response)
        return sql, response if not sql else None
    except Exception as e:
        return None, f"SQL generation failed: {str(e)}"

def execute_query(sql):
    try:
        result = session.execute(text(sql)).fetchall()
        return result if result else "✅ Query ran but returned no rows."
    except Exception as e:
        return f"❌ SQL execution error: {str(e)}"

def summarize_result(prompt, results):
    """Use the summarizer agent to explain what the results mean"""
    try:
        results_str = str(results)
        summary_prompt = (
            f"User question: {prompt}\n\n"
            f"Query result: {results_str}\n\n"
            f"Based on the schema provided earlier, summarize this result in a helpful way."
        )
        response = summarizer_agent.run(summary_prompt)
        return getattr(response, "content", str(response)).strip()
    except Exception as e:
        return f"❌ Summarization error: {str(e)}"

# --- Main ---
if __name__ == "__main__":
    user_prompt = input("Enter your query: ")

    if not validate_prompt(user_prompt):
        print("⚠️ Your prompt is not related to the flight booking system.")
    else:
        sql_query, fallback = generate_sql(user_prompt)
        if sql_query:
            print(f"\n✅ Generated SQL:\n{sql_query}")
            results = execute_query(sql_query)
            print(f"\n📊 Query Results:\n{results}")
            summary = summarize_result(user_prompt, results)
            print(f"\n📝 Summary:\n{summary}")
        else:
            print("\n⚠️ Could not extract SQL. Here is the agent's response:\n")
            print(fallback or "No response.")

    session.close()
