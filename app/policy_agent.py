import json
import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from dotenv import load_dotenv

# Importable both as `streamlit run policy_agent.py` from app/ and as app/policy_agent.py from root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()
# Streamlit Cloud supplies config via st.secrets; the RAG modules read os.environ at import time.
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass  # no secrets.toml (local runs use .env)
# ----------------------------
# MCP + Agent Configuration
# ----------------------------
# Streamlit Cloud can't reach localhost — set MCP_URL in st.secrets to your public tunnel.
DEFAULT_MCP_URL = os.getenv("MCP_URL", "http://localhost:8000/mcp")

SYSTEM_PROMPT = """You are an HR Expense Compliance Agent.
Process each claim in these steps:
1) First, use the rag_ask tool to get the company expense policy by passing questions and category "policies"
2) Evaluate the claim strictly against the retrieved policy
3) Based on your evaluation:
   - If compliant: use the approve tool
   - If non-compliant: use the reject tool
4) Then provide your decision in JSON format
You MUST use the appropriate tool (approve or reject) based on your evaluation.
"""

ASK_TEMPLATE = """Given the company expense policy, decide the outcome for this claim.

Return strictly JSON:
{{
  "decision": "approve" | "reject",
  "reason": "<one-sentence reason appropriate for the policy>",
  "violated_clause": "<optional: cite the specific clause if rejecting in detail>"
}}

Claim:
•⁠  ⁠claim_id: {claim_id}
•⁠  ⁠date: {date}
•⁠  ⁠category: {category}
•⁠  ⁠description: {description}
•⁠  ⁠amount: {amount} {currency}
•⁠  ⁠receipt_available: {receipt_available}
•⁠  ⁠pre_approved: {pre_approved}
"""

# ----------------------------
# Helpers
# ----------------------------
EXTRACT_PROMPT = """Extract every expense claim in this document as strict JSON matching:

{"employee": {"id": ..., "name": ..., "department": ..., "designation": ..., "location": ...},
 "claims": [{"claim_id": ..., "date": ..., "category": ..., "description": ...,
             "amount": ..., "currency": ..., "receipt_available": ..., "pre_approved": ...}]}

amount is a number, receipt_available and pre_approved are booleans, date is YYYY-MM-DD.
Use null for anything the document does not state. Return only the JSON object.

Document:
"""


def load_claims_from_bytes(file_bytes: bytes) -> Dict[str, Any]:
    return json.loads(file_bytes.decode("utf-8"))


def extract_claims_from_document(file_bytes: bytes, filename: str, model: str) -> Dict[str, Any]:
    """PDF/Word claim doc -> the same {employee, claims} dict a claims JSON would give."""
    from app.ingest import load_file

    suffix = Path(filename).suffix.lower()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        docs = load_file(tmp_path, "claims")
    finally:
        if tmp_path:
            os.unlink(tmp_path)

    text = "\n\n".join(d.page_content for d in docs).strip()
    if not text:
        raise ValueError(f"No text could be read from {filename}")

    llm = ChatOpenAI(model=model, temperature=0).bind(
        response_format={"type": "json_object"}
    )
    return json.loads(llm.invoke(EXTRACT_PROMPT + text[:20000]).content)

async def build_agent(mcp_url: str, model: str = "gpt-4o-mini", temperature: float = 0.0):
    client=MultiServerMCPClient(
        {
            "tools":{
                "url":mcp_url,
                "transport":"streamable_http"
            }
        }
    )
    tools=await client.get_tools()
    llm=ChatOpenAI(model=model,temperature=temperature)
    agent=create_agent(llm,tools)
    return agent 

async def process_claims(agent, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    employee = data.get("employee", {})
    claims: List[Dict[str, Any]] = data.get("claims", [])

    results = []
    employee_ctx = f"{employee.get('department','')}, {employee.get('designation','')}, {employee.get('location','')}"

    for claim in claims:
        # Build the user content (policy evaluation prompt)
        user_content = ASK_TEMPLATE.format(
            employee_id=employee.get("id"),
            claim_id=claim.get("claim_id"),
            date=claim.get("date"),
            category=claim.get("category"),
            description=claim.get("description"),
            amount=claim.get("amount"),
            currency=claim.get("currency"),
            receipt_available=claim.get("receipt_available"),
            pre_approved=claim.get("pre_approved"),
            employee_context=employee_ctx,
        )

        # Pass both system and user messages so the agent knows to approve/reject after reading policy
        response = await agent.ainvoke({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
        })

        try:
            final_msg = response["messages"][-1].content
        except Exception:
            final_msg = str(response)

        results.append({
            "claim_id": claim.get("claim_id"),
            "category": claim.get("category"),
            "amount": f"{claim.get('amount')} {claim.get('currency')}",
            "decision_trace": final_msg,
        })

    return results

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="HR Expense Agent (MCP)", page_icon="💼", layout="wide",
                   initial_sidebar_state="expanded")
st.title("💼 HR Expense Compliance Agent (MCP + RAG)")

with st.sidebar:
    st.header("Settings")
    mcp_url = st.text_input("MCP Server URL", value=DEFAULT_MCP_URL)
    model = st.text_input("OpenAI Model", value="gpt-4o-mini")
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.0, 0.1)
    st.caption("MCP server should expose tools: rag_ask, approve, reject.")

    st.divider()
    st.header("Knowledge base")
    kb_file = st.file_uploader("Upload policy doc", type=["pdf", "docx", "md", "txt"])
    kb_category = st.text_input("Category", value="policies",
                                help="Becomes the retrieval filter; 'policies' is what the agent queries.")
    if st.button("Ingest into RAG", use_container_width=True):
        if not kb_file:
            st.error("Pick a file first.")
        else:
            with st.spinner(f"Ingesting {kb_file.name}..."):
                try:
                    # Imported lazily: needs DATABASE_URL/OPENAI_API_KEY at import time,
                    # so a missing secret fails here instead of blanking the whole page.
                    from app.ingest import ingest_file_async
                    from app.uploads import safe_dest

                    dest = safe_dest(kb_file.name, kb_category.strip())
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(kb_file.getvalue())
                    stats = asyncio.run(ingest_file_async(str(dest), dest.parent.name))
                except Exception as e:
                    st.error(f"Ingest failed: {e}")
                else:
                    st.success(f"Ingested {dest.name} → {stats['chunks']} chunks into '{dest.parent.name}'")

uploaded = st.file_uploader("Upload claims (JSON, PDF or Word)", type=["json", "pdf", "docx"],
                            help="PDF/Word claim documents are read and converted to claims automatically.")

col1, col2 = st.columns([1, 1])
with col1:
    run_btn = st.button("Run Agent on Claims", type="primary", use_container_width=True)
with col2:
    st.write("")

if run_btn:
    if not uploaded:
        st.error("Please upload a claims JSON file first.")
        st.stop()

    is_json = Path(uploaded.name).suffix.lower() == ".json"
    try:
        if is_json:
            data = load_claims_from_bytes(uploaded.getvalue())
        else:
            with st.spinner(f"Reading claims out of {uploaded.name}..."):
                data = extract_claims_from_document(uploaded.getvalue(), uploaded.name, model)
    except Exception as e:
        st.error(f"Failed to read {uploaded.name}: {e}")
        st.stop()

    if not data.get("claims"):
        st.error(f"No claims found in {uploaded.name}.")
        st.stop()

    employee = data.get("employee", {})
    st.subheader("Employee")
    st.json(employee)

    if not is_json:
        with st.expander(f"Claims extracted from {uploaded.name} — check before running", expanded=True):
            st.json(data.get("claims"))

    with st.spinner("Connecting to MCP server and building agent..."):
        try:
            agent = asyncio.run(build_agent(mcp_url=mcp_url, model=model, temperature=temperature))
        except Exception as e:
            st.error(f"Failed to connect/build agent: {e}")
            st.stop()

    st.info(f"Found {len(data.get('claims', []))} claim(s). Running decisions + actions...")
    with st.spinner("Evaluating claims and taking actions..."):
        try:
            results = asyncio.run(process_claims(agent, data))
        except Exception as e:
            st.error(f"Agent run failed: {e}")
            st.stop()

    st.success("Done!")
    st.subheader("Results")
    for r in results:
        with st.expander(f"Claim {r['claim_id']} — {r['category']} — {r['amount']}", expanded=False):
            st.markdown("Decision / Action Trace")
            st.write(r["decision_trace"])

    st.caption(
        "The agent evaluates each claim against policy, then finalizes it as approved or rejected. "
        "Your MCP server performs the actual actions via its tools."
    )